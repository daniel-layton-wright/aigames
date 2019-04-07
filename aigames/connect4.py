import copy

import numpy as np

from aigames.base.game import *


class Connect4(SequentialGame):
    N_PLAYERS = 2
    EMPTY = 0
    WIN_REWARD = 1
    LOSE_REWARD = -1
    ILLEGAL_ACTION_PENALTY = -2

    STATE_SIZE = (7*6)*3
    STATE_SHAPE = (3, 6, 7)
    ACTION_SIZE = 1
    ALL_ACTIONS = list(range(7))

    def __init__(self, players, verbose = False, pause_seconds = 0, debugger = None):
        # Save input
        self.players = players
        self.verbose = verbose
        self.pause_seconds = pause_seconds
        self.debugger = debugger

        # Initialize the board
        self.state = np.zeros((3,6,7)).astype(int)

    @staticmethod
    def get_player_index(state):
        return state[2,0,0]

    @staticmethod
    def legal_actions(state):
        open_indices = sorted(list(zip(*np.where(state[:2,:,:].sum(axis = 0).sum(axis = 0) < 6))))
        # these are now tuples like [(0,), (1,), ...]  --> convert to [0, 1, ...]
        return [x[0] for x in open_indices]

    @staticmethod
    def is_terminal_state(state):
        if state[:2,:,:].sum() == 42:
            return True

        for mask in Connect4.masks:
            h, w = mask.shape
            for player_index in range(Connect4.N_PLAYERS):
                player_board = state[player_index,:,:]
                for y in range(player_board.shape[0] - h + 1):
                    for x in range(player_board.shape[1] - w + 1):
                        if (player_board[y:(y+h), x:(x+w)] * mask).sum() == mask.sum():
                            return True

        return False

    @staticmethod
    def get_next_state(state, action):
        new_state = copy.deepcopy(state)
        cur_player_index = Connect4.get_player_index(state)

        heights = state[:2,:,:].sum(axis = 0).sum(axis = 0)
        height = state.shape[1] - heights[action] - 1

        new_state[cur_player_index][(height,action)] = 1
        new_state[2,:,:] = 1 - state[2,:,:]  # change the player index
        return new_state

    @staticmethod
    def reward(state, player_index):
        for mask in Connect4.masks:
            h, w = mask.shape
            for cur_player_index in range(Connect4.N_PLAYERS):
                player_board = state[cur_player_index,:,:]
                for y in range(player_board.shape[0] - h + 1):
                    for x in range(player_board.shape[1] - w + 1):
                        if (player_board[y:(y+h), x:(x+w)] * mask).sum() == mask.sum():
                            if cur_player_index == player_index:
                                return Connect4.WIN_REWARD
                            else:
                                return Connect4.LOSE_REWARD

        return 0

    def __str__(self):
        return self.state_to_str(self.state)

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
            '-------------\n'.join(['|'.join(line_to_marks(state[:,i,:])) + '\n' for i in
                            range(state.shape[1])])
        )

    masks = [
        np.array([
            [1],
            [1],
            [1],
            [1]
        ]),
        np.array([
            [1, 1, 1, 1]
        ]),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),
        np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ])
    ]