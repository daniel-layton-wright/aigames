from ..game.game import GameListener
from ..game.game_multi import GameListenerMulti
from ..game import SequentialGame
from tqdm.auto import tqdm
import torch


class RewardListener(GameListener):
    def __init__(self, discount_rate, player_index, show_tqdm=False, tqdm_total=None):
        self.reward = 0
        self.i = 0
        self.discount_rate = discount_rate
        self.player_index = player_index

        self.show_tqdm = show_tqdm
        self.tqdm_total = tqdm_total
        self.tqdm = None

    def before_game_start(self, game):
        self.reward = 0
        self.i = 0

        if self.show_tqdm:
            self.tqdm = tqdm(total=self.tqdm_total) if self.show_tqdm else None
            self.tqdm.set_description('Game rewards')

    def on_action(self, game, action):
        next_state, rewards = game.get_next_state_and_rewards(game.state, action)
        reward_increment = (self.discount_rate ** self.i) * rewards[self.player_index]
        self.reward += reward_increment
        self.i += 1
        self.tqdm.update(reward_increment)


class AvgRewardListenerMulti(GameListenerMulti):
    def __init__(self, discount_rate, player_index, show_tqdm=False, tqdm_total=None):
        self.rewards = None
        self.i = 0
        self.discount_rate = discount_rate
        self.player_index = player_index

        self.show_tqdm = show_tqdm
        self.tqdm_total = tqdm_total
        self.tqdm = None

    def before_game_start(self, game):
        self.rewards = torch.zeros(game.states.shape[0], dtype=torch.float32, device=game.states.device)

        if self.show_tqdm:
            self.tqdm = tqdm(total=self.tqdm_total) if self.show_tqdm else None
            self.tqdm.set_description('Game rewards')

    def on_rewards(self, rewards):
        reward_increment = (self.discount_rate ** self.i) * rewards[:, self.player_index]
        self.rewards += reward_increment
        self.i += 1
        self.tqdm.update((reward_increment.sum() / self.rewards.shape[0]).cpu().item())


class ActionCounterProgressBar(GameListenerMulti):
    def __init__(self, progress_bar_max, description='Game action count'):
        self.progress_bar_max = progress_bar_max
        self.tqdm = None
        self.description = description

    def before_game_start(self, game):
        self.tqdm = tqdm(total=self.progress_bar_max, leave=False)
        self.tqdm.set_description(self.description)

    def on_action(self, game, actions):
        self.tqdm.update(1)

    def on_game_end(self, game):
        self.tqdm.close()
        self.tqdm = None


class AverageRewardListener(RewardListener):
    def __init__(self, discount_rate, player_index):
        super().__init__(discount_rate, player_index)
        self.n_games = 0
        self.total_reward = 0
        self.avg_reward = 0

    def on_game_end(self, game):
        self.n_games += 1
        self.total_reward += self.reward
        self.avg_reward = self.total_reward / self.n_games


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
