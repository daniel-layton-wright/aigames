

class Agent:
    def get_action(self, state, legal_actions) -> int:
        raise NotImplementedError()

    def on_reward(self, reward, next_state, player_index):
        pass

    def before_game_start(self, n_players):
        pass

    def on_game_end(self):
        pass

    def on_action(self, state, action, next_state):
        pass
