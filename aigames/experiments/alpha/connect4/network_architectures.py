from functools import lru_cache
from torch import nn
from aigames import Flatten
from aigames.training_manager.alpha_training_manager import AlphaNetworkEvaluator
from aigames.game.connect4 import Connect4State, Connect4BitState
import torch
from torchvision.models.resnet import BasicBlock


class Connect4Evaluator(AlphaNetworkEvaluator):
    def process_state(self, state: Connect4State):
        s = state.grid
        if abs(s).sum() % 2 == 1:
            s = -1 * s

        t = torch.zeros(2, 6, 7, dtype=torch.float)
        t[0, :, :] = s

        t[1, :, :] = (t[0, :, :] == -1)
        t[0, :, :] += t[1, :, :]

        return t


class Connect4Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_channels1 = 64
        self.base_out_features = 16

        self.base = nn.Sequential(
            nn.ConstantPad2d((0, 0, 1, 0), 0),
            nn.Conv2d(in_channels=2, out_channels=self.n_channels1, kernel_size=7, stride=1),
            nn.BatchNorm2d(self.n_channels1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.n_channels1, out_channels=self.base_out_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.base_out_features),
            nn.ReLU(),
            Flatten()
        )

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=7),
            nn.Softmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=1)
        )

    def forward(self, processed_state):
        base = self.base(processed_state)
        policy = self.policy_head(base).squeeze()
        value = self.value_head(base)
        return policy, value


class Connect4EvaluatorV2(AlphaNetworkEvaluator):
    def process_state(self, state: Connect4BitState):
        return state.tensor_state

    @lru_cache(maxsize=100000)
    def evaluate(self, state):
        return super().evaluate(state)


class Connect4NetworkV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_channels1 = 128
        self.base_out_features = 32

        self.base = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=self.n_channels1, kernel_size=7, stride=1),
            nn.BatchNorm2d(self.n_channels1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.n_channels1, out_channels=self.base_out_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.base_out_features),
            nn.ReLU(),
            Flatten()
        )

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=7),
            nn.Softmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=1)
        )

    def forward(self, processed_state):
        base = self.base(processed_state)
        policy = self.policy_head(base).squeeze()
        value = self.value_head(base)
        return policy, value


class Connect4NetworkV3(nn.Module):
    def __init__(self, n_blocks=4, n_channels=64, n_out_channels=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_blocks = n_blocks

        self.base = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            *[BasicBlock(n_channels, n_channels) for _ in range(n_blocks)],
            nn.Conv2d(in_channels=n_channels, out_channels=n_out_channels, kernel_size=1, stride=1, padding=0),
            Flatten()
        )

        self.n_base_out_features = n_out_channels * 7 * 7

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.n_base_out_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=7),
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
