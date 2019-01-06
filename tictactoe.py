import numpy as np
import copy
from itertools import product
from game import *


class TicTacToe(SequentialGame):
    EMPTY = 0
    PLAYER_MARKS = [1, -1]
    WIN_REWARD = 1
    LOSE_REWARD = -1
    ILLEGAL_ACTION_PENALTY = -2

    STATE_SIZE = 9
    ALL_ACTIONS = list(product(range(3), range(3)))

    def __init__(self, players, verbose = False, pause_seconds = 0, debugger = None):
        # Save input
        self.players = players
        self.verbose = verbose
        self.pause_seconds = pause_seconds
        self.debugger = debugger

        # Initialize the board
        self.state = self.EMPTY * np.ones((3,3))

    @staticmethod
    def get_player_index(state):
        num_p0 = len(list(zip(*np.where(state == TicTacToe.PLAYER_MARKS[0]))))
        num_p1 = len(list(zip(*np.where(state == TicTacToe.PLAYER_MARKS[1]))))
        if num_p0 > num_p1:
            return 1
        else:
            return 0

    @staticmethod
    def legal_actions(state):
        return list(zip(*np.where(state == TicTacToe.EMPTY)))

    @staticmethod
    def is_terminal_state(state):
        return any((mask * state == TicTacToe.PLAYER_MARKS[0]).sum() == 3
                   or (mask * state == TicTacToe.PLAYER_MARKS[1]).sum() == 3
                   for mask in TicTacToe.masks) \
               or (state != TicTacToe.EMPTY).all()

    @staticmethod
    def get_next_state(state, action):
        new_state = copy.deepcopy(state)
        new_state[action] = TicTacToe.PLAYER_MARKS[TicTacToe.get_player_index(state)]
        return new_state

    @staticmethod
    def reward(state, player_index):
        other_player_index = 1 if player_index == 0 else 0
        if any((mask * state == TicTacToe.PLAYER_MARKS[player_index]).sum() == 3 for mask in TicTacToe.masks):
            return TicTacToe.WIN_REWARD
        elif any((mask * state == TicTacToe.PLAYER_MARKS[other_player_index]).sum() == 3 for mask in TicTacToe.masks):
            return TicTacToe.LOSE_REWARD
        else:
            return 0

    def __str__(self):
        return self.state_to_str(self.state)

    @staticmethod
    def state_to_str(state):
        def line_to_marks(line):
            marks = {
                TicTacToe.PLAYER_MARKS[0]: 'x',
                TicTacToe.PLAYER_MARKS[1]: 'o',
                TicTacToe.EMPTY: ' '
            }
            return list(map(lambda x: marks[x], line))

        return (
            '-----\n'.join(['{}|{}|{}\n'.format(*line_to_marks(state[i, :])) for i in
                            range(state.shape[0])])
        )

    masks = [
        np.array([
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ]),
        np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ]),
        np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1]
        ]),
        np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0]
        ]),
        np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ]),
        np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]),
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
    ]