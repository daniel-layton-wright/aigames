from functools import lru_cache
from torch import nn
from aigames import Flatten
from aigames.training_manager.alpha_training_manager import AlphaNetworkEvaluator
from aigames.game.twenty_forty_eight import TwentyFortyEight, TwentyFortyEightState
import torch
from torchvision.models.resnet import BasicBlock


class TwentyFortyEightEvaluator(AlphaNetworkEvaluator):
    def process_state(self, state: TwentyFortyEightState):
        return torch.FloatTensor(state.grid).unsqueeze(0)


class TwentyFortyEightNetwork(nn.Module):
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
        value = self.value_head(base)
        return policy, value