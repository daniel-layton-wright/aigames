from aigames.base.agent import *


class ManualAgent(SequentialAgent):
    def __init__(self, game = None, mode = 'eval'):
        super().__init__(game)
        self.mode = mode

    def choose_action(self, state, player_index, verbose = False):
        choice = input('Enter the action to play: ')
        if self.mode == 'eval':
            choice = eval(choice)
        elif self.mode == 'index':
            choice = self.game.legal_actions(state)[int(choice)]

        return choice

    def reward(self, reward_value, state, i):
        pass