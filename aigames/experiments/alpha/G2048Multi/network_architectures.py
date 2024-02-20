from typing import Tuple
import torch
from torch import nn
from ....agent.alpha_agent import BaseAlphaEvaluator
from ....training_manager.alpha_training_manager import AlphaNetworkEvaluator
from ....training_manager.alpha_training_manager_multi_lightning import AlphaMultiNetwork
from torchvision.models.resnet import BasicBlock
from .... import Flatten


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
    def __init__(self, n_blocks=4, n_channels=64, n_out_channels=32, n_value_buckets=251, bucket_min=0, bucket_max=250,
                 value_scale_epsilon=1e-3):
        super().__init__()
        self.n_channels = n_channels
        self.n_blocks = n_blocks
        self.n_value_buckets = n_value_buckets
        self.buckets = torch.linspace(bucket_min, bucket_max, n_value_buckets)
        self.value_scale_epsilon = value_scale_epsilon

        self.base = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=n_channels, kernel_size=4, stride=1, padding=0),
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
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.n_value_buckets),
            nn.Softmax(dim=1)
        )

    def forward(self, processed_state):
        base = self.base(processed_state)
        policy = self.policy_head(base).squeeze()
        value_bucket_softmax = self.value_head(base)
        return policy, value_bucket_softmax

    def evaluate(self, state):
        pi, value_bucket_softmax = self.forward(self.process_state(state))
        scaled_value = self.digitize(value_bucket_softmax)
        value = self.inverse_scale(scaled_value).unsqueeze(1)
        return pi, value

    def process_state(self, state):
        return torch.eye(16, dtype=torch.float32, device=state.device)[state.to(torch.long)].transpose(1, 3).transpose(2, 3)

    def scale_value(self, value):
        return torch.sign(value) * (torch.sqrt(torch.abs(value) + 1) - 1 + self.value_scale_epsilon * value)

    def bucketize(self, scaled_value):
        bucketized = torch.clip(torch.bucketize(scaled_value, self.buckets), max=(self.n_value_buckets-1))
        bucketized_low = torch.clip(bucketized - 1, min=0)
        bucket_values = self.buckets[bucketized]
        one_less_buckets = self.buckets[bucketized_low]
        bucket_weight = torch.clip((scaled_value - one_less_buckets)
                                   / (bucket_values - one_less_buckets + (bucket_values == one_less_buckets)),
                                   min=0.0, max=1.0)
        out = torch.zeros((scaled_value.shape[0], self.n_value_buckets), device=scaled_value.device)
        out[torch.arange(scaled_value.shape[0]), bucketized.flatten()] = bucket_weight.flatten()
        out[torch.arange(scaled_value.shape[0]), bucketized_low.flatten()] = 1 - bucket_weight.flatten()
        return out

    def digitize(self, bucketized_values):
        return torch.sum(self.buckets * bucketized_values, dim=1)

    def scale_and_bucketize(self, value):
        value = value.reshape(-1, 1)
        scaled_value = self.scale_value(value)
        out = self.bucketize(scaled_value)
        return out

    def inverse_scale(self, scaled_value):
        hp = 1 + torch.abs(scaled_value)
        e = self.value_scale_epsilon
        sgn = torch.sign(scaled_value)

        return (scaled_value != 0).float() * (
                hp/e
                + sgn*1/(2*e*e)
                - sgn*torch.sqrt(sgn*hp/e + 1/(4*e*e) + 1)/e)

    def loss(self, states, pis, values) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_distns, pred_values = self.forward(states)

        nan_distns = torch.isnan(pis).any(dim=1)

        values = self.scale_and_bucketize(values)

        value_loss = (-torch.sum(values * torch.log(pred_values), dim=1)).mean()
        distn_loss = (-torch.sum(pis[~nan_distns] * torch.log(pred_distns[~nan_distns]), dim=1)).mean()
        return value_loss, distn_loss
