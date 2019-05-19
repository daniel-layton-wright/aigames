class Agent:
    pass

class SequentialAgent(Agent):
    def __init__(self, game_class):
        self.game_class = game_class

    def choose_action(self, state, player_index, verbose = False):
        raise NotImplementedError()

    def reward(self, reward_value, state, player_index):
        raise NotImplementedError()

    def start_episode(self):
        pass

    def end_episode(self):
        pass
