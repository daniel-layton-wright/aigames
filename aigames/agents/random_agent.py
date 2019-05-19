import numpy as np
from aigames.base.agent import *


class RandomAgent(SequentialAgent):
    def __init__(self, game_class):
        super().__init__(game_class)

    def choose_action(self, state, player_index, verbose = False):
        legal_actions = self.game_class.legal_actions(state)
        random_idx = np.random.choice(len(legal_actions))
        return legal_actions[random_idx]

    def reward(self, reward_value, state, player_index):
        pass
