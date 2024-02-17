from typing import Type
from .agent import AgentMulti
from ..game.game_multi import GameMulti
import torch


class RandomAgentMulti(AgentMulti):
    def __init__(self, game_class: Type[GameMulti]):
        super().__init__()
        self.game_class = game_class
        self.n_actions = game_class.get_n_actions()

    def get_actions(self, states, mask):
        legal_actions = self.game_class.get_legal_action_masks(states)
        return torch.multinomial(legal_actions, num_samples=1)
