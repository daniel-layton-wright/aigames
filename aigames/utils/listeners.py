from ..game.game import GameListener
from ..game import SequentialGame


class RewardListener(GameListener):
    def __init__(self, discount_rate, player_index):
        self.reward = 0
        self.i = 0
        self.discount_rate = discount_rate
        self.player_index = player_index

    def before_game_start(self, game):
        self.reward = 0
        self.i = 0

    def on_action(self, game, action):
        next_state, rewards = game.get_next_state_and_rewards(game.state, action)
        self.reward += (self.discount_rate ** self.i) * rewards[self.player_index]
        self.i += 1


class GameHistoryListener(GameListener):
    """
    Stores the history of a game. And resets when a new game starts.
    """
    def __init__(self):
        self.history = []

    def before_game_start(self, game: SequentialGame):
        self.history = []
        self.history.append(game.state)

    def after_action(self, game: SequentialGame):
        self.history.append(game.state)

    def __repr__(self):
        out = ''
        for state in self.history:
            out += str(state)
            out += '\n\n'

        return out
