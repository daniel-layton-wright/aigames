class Agent:
    pass

class SequentialAgent(Agent):
    def choose_action(self, state, player_index):
        raise NotImplementedError()

    def reward(self, reward_value, state, player_index):
        raise NotImplementedError()