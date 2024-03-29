from .agent import *
from .game import *
from .training_manager import *
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
