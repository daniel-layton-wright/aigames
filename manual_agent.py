from agent import *


class ManualTTTAgent(SequentialAgent):
    def __init__(self):
        pass

    def choose_action(self, state, player_index, verbose = False):
        choice = input('Enter the row and column to play: ')
        row, col = list(map(int, choice.split(',')))
        row -= 1
        col -= 1
        return (row, col)

    def reward(self, reward_value, state, i):
        pass