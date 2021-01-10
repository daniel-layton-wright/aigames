

class Agent:
    def get_action(self, state, legal_actions) -> int:
        raise NotImplementedError()
    
    def on_reward(self, reward, next_state):
        pass

    def before_game_start(self):
        pass
