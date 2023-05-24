from ..game.game import GameListener


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