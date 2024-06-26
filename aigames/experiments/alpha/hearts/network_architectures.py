from dataclasses import dataclass
from typing import Tuple
from aigames.agent.alpha_agent_multi import BaseAlphaEvaluator
from aigames.game.hearts import Hearts
from aigames.training_manager.alpha_training_manager_multi_lightning import AlphaMultiNetwork
import torch
from torch import nn

from aigames.utils.utils import digitize, bucketize


@dataclass(frozen=True, kw_only=True)
class HeartsNetworkHyperparameters:
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = 'ReLU'
    n_value_buckets: int = 27


class AttentionBlock(nn.Module):
    def __init__(self, config: HeartsNetworkHyperparameters):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3)
        self.multihead_attn = nn.MultiheadAttention(config.embed_dim, config.num_heads, config.dropout,
                                                    batch_first=True)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        out = self.multihead_attn(q, k, v, need_weights=False)[0]
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: HeartsNetworkHyperparameters):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = AttentionBlock(config)
        self.resid_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Use the activation function specified by the config
        activation = getattr(nn, config.activation)

        self.ffwd = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            activation(),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.resid_dropout(self.resid_proj(self.attn(self.ln1(x))))
        x = x + self.ffwd(self.ln2(x))
        return x


class HeartsNetwork(AlphaMultiNetwork, BaseAlphaEvaluator):
    """
    The HeartsNetwork uses a transformer architecture. There are embeddings for each of the values in the states,
    which all represent something discrete, such as a player index or an order index
    In addition there are positional embeddings for the order of the states in the sequence.

    """
    def __init__(self, hyperparameters: HeartsNetworkHyperparameters, game_class=Hearts):
        super().__init__()
        self.hyperparams = hyperparameters

        self.embeddings = nn.Embedding(num_embeddings=117, embedding_dim=hyperparameters.embed_dim)
        self.pos_embeddings = nn.Embedding(263, hyperparameters.embed_dim)
        self.row_embeddings = nn.Embedding(6, hyperparameters.embed_dim)
        self.card_embeddings = nn.Embedding(52, hyperparameters.embed_dim)
        self.embedding_dropout = nn.Dropout(hyperparameters.dropout)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(hyperparameters) for _ in range(hyperparameters.num_layers)])
        self.ln = nn.LayerNorm(hyperparameters.embed_dim)

        activation = getattr(nn, hyperparameters.activation)

        self.activation = activation()

        self.value_head = nn.Sequential(
            nn.Linear(hyperparameters.embed_dim, hyperparameters.embed_dim),
            activation(),
            nn.Linear(hyperparameters.embed_dim, self.hyperparams.n_value_buckets * game_class.get_n_players())
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hyperparameters.embed_dim, hyperparameters.embed_dim),
            activation(),
            nn.Linear(hyperparameters.embed_dim, game_class.get_n_actions())
        )

        flattened_state_mask = torch.ones((275,), dtype=torch.bool)
        flattened_state_mask[[1*55, 1*55+1, 1*55+2, 2*55, 2*55+1, 2*55+2, 3*55, 3*55+1, 3*55+2, 4*55,
                                   4*55+1, 4*55+2]] = False
        self.register_buffer('flattened_state_mask', flattened_state_mask)

        row_indices = torch.zeros((263,), dtype=torch.int)
        row_indices[3:55] = 1
        row_indices[55:107] = 2
        row_indices[107:159] = 3
        row_indices[159:211] = 4
        row_indices[211:] = 5
        self.register_buffer('row_indices', row_indices)

        card_indices = torch.zeros((263,), dtype=torch.int)
        card_indices[3:] = torch.arange(52).repeat(5)
        self.register_buffer('card_indices', card_indices)

        self.register_buffer('buckets', torch.linspace(-26, 0, self.hyperparams.n_value_buckets))

    def forward(self, states):
        """
        Output the logits for the policy and the value buckets
        """
        B, T = states.shape
        x = (self.embeddings(states) + self.pos_embeddings(torch.arange(T, device=states.device))
             + self.row_embeddings(self.row_indices))
        x[:, 3:] += self.card_embeddings(self.card_indices[states[:, 3:]])
        x = self.embedding_dropout(x)
        x = self.transformer_blocks(x)
        x = self.ln(x)
        x = x[:, 0, :]  # just take the first element of the sequence (arbitrary)
        x = self.activation(x)

        pi_logits = self.policy_head(x)
        value_bucket_logits = self.value_head(x).reshape(B, -1, self.hyperparams.n_value_buckets)

        return pi_logits, value_bucket_logits

    @torch.no_grad()
    def evaluate(self, state):
        if state.shape[1] != 275:
            state = self.process_state(state)

        pi_logits, value_bucket_logits = self.forward(state.to(self.device))

        pi = torch.softmax(pi_logits, dim=1)
        value_bucket_softmax = torch.softmax(value_bucket_logits, dim=-1)

        value = digitize(value_bucket_softmax, self.buckets).flatten(start_dim=1)
        return pi, value

    def process_state(self, state):
        state = state.flatten(start_dim=1)  # N x 5 x 55 -> N x 275

        # In the original state [:, 1-4, 0-2] are not used so let's filter those out
        state = state[:, self.flattened_state_mask].to(torch.int)

        # element 0 ranges 1-4 (player)
        # element 1 ranges 0-3 (count)
        # element 2 ranges 0-51 (count)
        # element 3-54 ranges 0-4 (player)
        # element 55-106 ranges 0-51 (order)
        # element 107-158 ranges 0-4 (player)
        # element 159-210 ranges 0-4 (player)
        # element 211-262 ranges 0-3 (order)
        state[:, 1] += 5
        state[:, 2] += (5 + 4)
        state[:, 55:107] += (5 + 4 + 52)
        state[:, 211:263] += (5 + 4 + 52 + 52)

        return state

    def loss(self, states, pis, values, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_distn_logits, pred_value_logits = self.forward(states)

        values = bucketize(values.flatten(), self.buckets, self.hyperparams.n_value_buckets)

        value_loss = nn.functional.cross_entropy(pred_value_logits.view(values.shape), values, reduction='none')
        distn_loss = nn.functional.cross_entropy(pred_distn_logits, pis, reduction='none')

        if len(args) > 0:
            importance_sampling_weights = args[0].to(pred_distn_logits.device)
            value_loss *= importance_sampling_weights
            distn_loss *= importance_sampling_weights

        value_loss = value_loss.mean()
        distn_loss = distn_loss.mean()

        return value_loss, distn_loss

    @classmethod
    def init_from_cfg(cls, cfg: HeartsNetworkHyperparameters):
        return cls(cfg)
