from dataclasses import dataclass
from typing import Tuple
import torch
from torch import nn
from ....agent.alpha_agent import BaseAlphaEvaluator
from ....training_manager.alpha_training_manager import AlphaNetworkEvaluator
from ....training_manager.alpha_training_manager_multi_lightning import AlphaMultiNetwork
from torchvision.models.resnet import BasicBlock
from .... import Flatten
from ....utils.utils import add_all_slots_to_arg_parser, load_from_arg_parser


class G2048MultiNetwork(AlphaMultiNetwork, AlphaNetworkEvaluator):
    def __init__(self, n_blocks=4, n_channels=64, n_out_channels=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_blocks = n_blocks

        self.base = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            *[BasicBlock(n_channels, n_channels) for _ in range(n_blocks)],
            nn.Conv2d(in_channels=n_channels, out_channels=n_out_channels, kernel_size=1, stride=1, padding=0),
            Flatten()
        )

        self.n_base_out_features = n_out_channels

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.n_base_out_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=4),
            nn.Softmax(dim=1)
        )

        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.n_base_out_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, processed_state):
        base = self.base(processed_state)
        policy = self.policy_head(base).squeeze()
        value = torch.exp(self.value_head(base))
        return policy, value

    def process_state(self, state: torch.Tensor):
        return state.unsqueeze(1)

    @torch.no_grad()
    def evaluate(self, state):
        # TODO : decide if need to do any device handling
        pi, v = self.forward(self.process_state(state))
        return pi,  v

    def loss(self, processed_states, action_distns, values):
        pred_distns, pred_values = self.forward(processed_states)

        nan_distns = torch.isnan(action_distns).any(dim=1)

        value_loss = ((values - pred_values) ** 2).sum(dim=1).mean()
        distn_loss = (-torch.sum(action_distns[~nan_distns] * torch.log(pred_distns[~nan_distns]), dim=1)).mean()
        return value_loss, distn_loss


class G2048MultiNetworkV2(AlphaMultiNetwork, BaseAlphaEvaluator):
    """
    This will use a categorical representation of the transformed value to train
    """

    @dataclass(kw_only=True, slots=True)
    class Hyperparameters:
        n_blocks: int = 4
        n_channels: int = 128
        n_out_channels: int = 128
        n_value_buckets: int = 251
        bucket_min: int = 0
        bucket_max: int = 250
        value_scale_epsilon: float = 1e-3

    def __init__(self, hyperparams: Hyperparameters = Hyperparameters()):
        super().__init__()
        self.hyperparams = hyperparams
        self.register_buffer('buckets', torch.linspace(self.hyperparams.bucket_min, self.hyperparams.bucket_max,
                                                       self.hyperparams.n_value_buckets))
        self.register_buffer('value_scale_epsilon', torch.FloatTensor([self.hyperparams.value_scale_epsilon]))

        self.base = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=self.hyperparams.n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hyperparams.n_channels),
            nn.ReLU(),
            *[BasicBlock(self.hyperparams.n_channels, self.hyperparams.n_channels) for _ in range(self.hyperparams.n_blocks)],
            nn.Conv2d(in_channels=self.hyperparams.n_channels, out_channels=self.hyperparams.n_out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.Flatten()
        )

        self.n_base_out_features = self.hyperparams.n_out_channels * 4 * 4

        self.policy_logits_head = nn.Sequential(
            nn.Linear(in_features=self.n_base_out_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=4)
        )

        self.value_logits_head = nn.Sequential(
            nn.Linear(in_features=self.n_base_out_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.hyperparams.n_value_buckets)
        )

    def forward(self, processed_state):
        base = self.base(processed_state)
        policy_logits = self.policy_logits_head(base)
        value_bucket_logits = self.value_logits_head(base)
        return policy_logits, value_bucket_logits

    @torch.no_grad()
    def evaluate(self, state):
        if state.shape[1] != 16:
            state = self.process_state(state)

        pi_logits, value_bucket_logits = self.forward(state)

        pi = torch.softmax(pi_logits, dim=1)
        value_bucket_softmax = torch.softmax(value_bucket_logits, dim=1)

        scaled_value = self.digitize(value_bucket_softmax, self.buckets)
        value = self.inverse_scale(scaled_value).unsqueeze(1)
        return pi, value

    def process_state(self, state):
        return torch.eye(16, dtype=torch.float32, device=state.device)[state.to(torch.long)].transpose(1, 3).transpose(2, 3)

    def scale_value(self, value):
        return torch.sign(value) * (torch.sqrt(torch.abs(value) + 1) - 1 + self.hyperparams.value_scale_epsilon * value)

    @staticmethod
    def bucketize(scaled_value, buckets, n_value_buckets):
        bucketized = torch.clip(torch.bucketize(scaled_value, buckets), max=(n_value_buckets-1))
        bucketized_low = torch.clip(bucketized - 1, min=0)
        bucket_values = buckets[bucketized]
        one_less_buckets = buckets[bucketized_low]
        bucket_weight = torch.clip((scaled_value - one_less_buckets)
                                   / (bucket_values - one_less_buckets + (bucket_values == one_less_buckets)),
                                   min=0.0, max=1.0)
        out = torch.zeros((scaled_value.shape[0], n_value_buckets), device=scaled_value.device)
        out[torch.arange(scaled_value.shape[0]), bucketized.flatten()] = bucket_weight.flatten()
        out[torch.arange(scaled_value.shape[0]), bucketized_low.flatten()] = 1 - bucket_weight.flatten()
        return out

    @staticmethod
    def digitize(bucketized_values, buckets):
        return torch.sum(buckets * bucketized_values, dim=1)

    def scale_and_bucketize_values(self, value):
        value = value.reshape(-1, 1)
        scaled_value = self.scale_value(value)
        out = self.bucketize(scaled_value, self.buckets, self.hyperparams.n_value_buckets)
        return out

    def inverse_scale(self, scaled_value):
        hp = 1 + torch.abs(scaled_value)
        e = self.hyperparams.value_scale_epsilon
        sgn = torch.sign(scaled_value)

        return (scaled_value != 0).float() * (
                hp/e
                + sgn*1/(2*e*e)
                - sgn*torch.sqrt(sgn*hp/e + 1/(4*e*e) + 1)/e)

    def loss(self, states, pis, values, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_distn_logits, pred_value_logits = self.forward(states)

        nan_distns = torch.isnan(pis).any(dim=1)

        values = self.scale_and_bucketize_values(values)

        value_loss = nn.functional.cross_entropy(pred_value_logits, values, reduction='none')
        distn_loss = nn.functional.cross_entropy(pred_distn_logits[~nan_distns], pis[~nan_distns],
                                                 reduction='none')

        if len(args) > 0:
            importance_sampling_weights = torch.tensor(args[0], device=pred_distn_logits.device)
            value_loss *= importance_sampling_weights
            distn_loss *= importance_sampling_weights[~nan_distns]

        value_loss = value_loss.mean()
        distn_loss = distn_loss.mean()

        return value_loss, distn_loss

    @classmethod
    def add_args_to_arg_parser(cls, parser):
        hypers = cls.Hyperparameters()
        add_all_slots_to_arg_parser(parser, hypers)

    @classmethod
    def init_from_arg_parser(cls, args):
        hypers = cls.Hyperparameters()
        load_from_arg_parser(args, hypers)

        return cls(hypers)


class G2048MultiNetworkV3(G2048MultiNetworkV2):
    @dataclass(kw_only=True, slots=True)
    class Hyperparameters(G2048MultiNetworkV2.Hyperparameters):
        n_num_move_buckets: int = 101
        num_move_bucket_min: int = 0
        num_move_bucket_max: int = 100

    def __init__(self, hyperparams: Hyperparameters = Hyperparameters()):
        super().__init__(hyperparams)
        self.hyperparams = hyperparams
        self.register_buffer('num_move_buckets',
                             torch.linspace(self.hyperparams.num_move_bucket_min, self.hyperparams.num_move_bucket_max,
                                            self.hyperparams.n_num_move_buckets))
        self.num_moves_head = nn.Sequential(
            nn.Linear(in_features=self.n_base_out_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.hyperparams.n_num_move_buckets)
        )

    def forward(self, processed_state):
        base = self.base(processed_state)
        policy_logits = self.policy_logits_head(base)
        value_bucket_logits = self.value_logits_head(base)
        num_moves_logits = self.num_moves_head(base)
        return policy_logits, value_bucket_logits, num_moves_logits

    def scale_and_bucketize_num_moves(self, num_moves):
        value = num_moves.reshape(-1, 1)
        scaled_value = self.scale_value(num_moves)
        out = self.bucketize(scaled_value, self.num_move_buckets, self.hyperparams.n_num_move_buckets)
        return out

    @torch.no_grad()
    def evaluate(self, state):
        if state.shape[1] != 16:
            state = self.process_state(state)

        pi_logits, value_bucket_logits, num_moves_logits = self.forward(state)

        pi = torch.softmax(pi_logits, dim=1)

        value_bucket_softmax = torch.softmax(value_bucket_logits, dim=1)
        scaled_value = self.digitize(value_bucket_softmax, self.buckets)
        value = self.inverse_scale(scaled_value).unsqueeze(1)

        num_moves_softmax = torch.softmax(num_moves_logits, dim=1)
        num_moves = self.digitize(num_moves_softmax, self.num_move_buckets)
        num_moves = self.inverse_scale(num_moves).unsqueeze(1)

        return pi, value, num_moves

    def loss(self, states, pis, values, num_moves, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_distn_logits, pred_value_logits, num_moves_logits = self.forward(states)

        values = self.scale_and_bucketize_values(values)
        value_loss = nn.functional.cross_entropy(pred_value_logits, values)

        nan_distns = torch.isnan(pis).any(dim=1)
        distn_loss = nn.functional.cross_entropy(pred_distn_logits[~nan_distns], pis[~nan_distns])

        num_moves = self.scale_and_bucketize_num_moves(num_moves)
        num_moves_loss = nn.functional.cross_entropy(num_moves_logits, num_moves)

        return value_loss, distn_loss, num_moves_loss
