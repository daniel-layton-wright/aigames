import copy
from itertools import product

import numpy as np

from aigames.base.game import *


class TicTacToe(SequentialGame):
    N_PLAYERS = 2
    EMPTY = 0
    WIN_REWARD = 1
    LOSE_REWARD = -1
    ILLEGAL_ACTION_PENALTY = -2

    STATE_SIZE = 9*3
    STATE_SHAPE = (3, 3, 3)
    ACTION_SIZE = 2
    ALL_ACTIONS = sorted(list(product(range(3), range(3))))

    def __init__(self, players, verbose = False, pause_seconds = 0, monitor = None):
        # Save input
        self.players = players
        self.verbose = verbose
        self.pause_seconds = pause_seconds
        self.monitor = monitor

        # Initialize the board
        self.state = np.zeros((3,3,3)).astype(int)

    @staticmethod
    def get_player_index(state):
        return state[2,0,0]

    @staticmethod
    def legal_actions(state):
        return sorted(list(set(zip(*np.where(state[0,:,:] == 0))).intersection(set(zip(*np.where(state[1,:,:] == 0))))))

    @staticmethod
    def is_terminal_state(state):
        return (
            any((mask * state[0, :, :] == 1).sum() == 3 for mask in TicTacToe.masks)
            or any((mask * state[1,:,:] == 1).sum() == 3 for mask in TicTacToe.masks)
            or state[:2,:,:].sum() == 9
        )

    @staticmethod
    def get_next_state(state, action):
        new_state = copy.deepcopy(state)
        cur_player_index = TicTacToe.get_player_index(state)
        new_state[cur_player_index][action] = 1
        new_state[2,:,:] = 1 - state[2,:,:]  # change the player index
        return new_state

    @staticmethod
    def reward(state, player_index):
        other_player_index = 1 if player_index == 0 else 0
        if any((mask * state[player_index,:,:] == 1).sum() == 3 for mask in TicTacToe.masks):
            return TicTacToe.WIN_REWARD
        elif any((mask * state[other_player_index,:,:] == 1).sum() == 3 for mask in TicTacToe.masks):
            return TicTacToe.LOSE_REWARD
        else:
            return 0

    def __str__(self):
        return self.state_to_str(self.state)

    @staticmethod
    def hashable_state(state):
        return tuple(state.flatten())

    @staticmethod
    def state_to_str(state):
        def line_to_marks(line):
            out = []
            for i in range(line.shape[1]):
                if line[0,i] == 1:
                    out.append('x')
                elif line[1,i] == 1:
                    out.append('o')
                else:
                    out.append(' ')

            return out

        return (
            '-----\n'.join(['{}|{}|{}\n'.format(*line_to_marks(state[:,i,:])) for i in
                            range(state.shape[1])])
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