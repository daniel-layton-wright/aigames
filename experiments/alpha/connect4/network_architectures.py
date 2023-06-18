from torch import nn
from aigames import Flatten


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
