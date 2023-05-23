from .alpha_networks import *
from .base import *
from .games import *
from .agents import *
from .monitors import *
from .play_against_minimax import *
from .train_utils import *
from .agent import *
from .game import *
from .training_manager import *
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

