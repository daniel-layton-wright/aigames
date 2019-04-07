import torch.nn as nn
import torch


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class AlphaAgentNetwork1(nn.Module):
    def __init__(self, game_class):
        super().__init__()
        self.game_class = game_class
        self.base = nn.Sequential(
            nn.Conv2d(in_channels = game_class.STATE_SHAPE[0], out_channels = 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.base_out_features = 32 * game_class.STATE_SHAPE[1] * game_class.STATE_SHAPE[2]

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=len(game_class.ALL_ACTIONS)),
            nn.Softmax()
        )
        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=1),
            nn.Tanh()
        )

    def forward(self, state):
        state_processed = self.process_state(state)
        base = self.base(state_processed)
        policy = self.policy_head(base).squeeze()
        value = self.value_head(base).squeeze()
        return policy, value

    @staticmethod
    def process_state(state):
        x = torch.FloatTensor(state)
        while len(x.shape) < 4:
            x = x.unsqueeze(0)

        return x


class AlphaAgentNetwork2(nn.Module):
    def __init__(self, game_class):
        super().__init__()
        self.game_class = game_class
        self.base = nn.Sequential(
            nn.Conv2d(in_channels = game_class.STATE_SHAPE[0], out_channels = 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            Flatten()
        )

        self.base_out_features = 32 * game_class.STATE_SHAPE[1] * game_class.STATE_SHAPE[2]

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=len(game_class.ALL_ACTIONS)),
            nn.Softmax()
        )
        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=1),
            nn.Tanh()
        )

    def forward(self, state):
        state_processed = self.process_state(state)
        base = self.base(state_processed)
        policy = self.policy_head(base).squeeze()
        value = self.value_head(base).squeeze()
        return policy, value

    @staticmethod
    def process_state(state):
        x = torch.FloatTensor(state)
        while len(x.shape) < 4:
            x = x.unsqueeze(0)

        return x


class AlphaAgentNetwork3(nn.Module):
    def __init__(self, game_class):
        super().__init__()
        self.game_class = game_class
        self.base = nn.Sequential(
            nn.Conv2d(in_channels = game_class.STATE_SHAPE[0], out_channels = 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.base_out_features = 8 * game_class.STATE_SHAPE[1] * game_class.STATE_SHAPE[2]

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=len(game_class.ALL_ACTIONS)),
            nn.Softmax()
        )
        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=1),
            nn.Tanh()
        )

    def forward(self, state):
        state_processed = self.process_state(state)
        base = self.base(state_processed)
        policy = self.policy_head(base).squeeze()
        value = self.value_head(base).squeeze()
        return policy, value

    @staticmethod
    def process_state(state):
        x = torch.FloatTensor(state)
        while len(x.shape) < 4:
            x = x.unsqueeze(0)

        return x


class AlphaAgentNetwork4(nn.Module):
    def __init__(self, game_class):
        super().__init__()
        self.game_class = game_class
        self.base = nn.Sequential(
            nn.Conv2d(in_channels = game_class.STATE_SHAPE[0], out_channels = 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.base_out_features = 1 * game_class.STATE_SHAPE[1] * game_class.STATE_SHAPE[2]

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=len(game_class.ALL_ACTIONS)),
            nn.Softmax()
        )
        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=1),
            nn.Tanh()
        )

    def forward(self, state):
        state_processed = self.process_state(state)
        base = self.base(state_processed)
        policy = self.policy_head(base).squeeze()
        value = self.value_head(base).squeeze()
        return policy, value

    @staticmethod
    def process_state(state):
        x = torch.FloatTensor(state)
        while len(x.shape) < 4:
            x = x.unsqueeze(0)

        return x