from typing import List
from .game import SequentialGame
import numpy as np
import torch
import copy
from scipy import sparse


class SparseDict:
    class Sentinel:
        pass

    def __init__(self, default_value=None):
        self.data = {}
        self.default_value = default_value

    def getcreate(self, item):
        out = self.data.get(item, self.Sentinel)
        if out == self.Sentinel:
            out = self.get_default()
            self.data[item] = out

        return out

    def get_default(self):
        return self.default_value if not callable(self.default_value) else self.default_value()

    def __getitem__(self, item):
        return self.data.get(item, self.get_default())

    def __setitem__(self, key, value):
        self.data[key] = value


class Connect4State:
    N_COLS = 7
    N_ROWS = 6

    def __init__(self):
        self.grid = torch.FloatTensor(np.zeros((self.N_ROWS, self.N_COLS))).unsqueeze(0)
        # self.neighbors = np.zeros((self.N_ROWS, self.N_COLS, 3, 3)).astype(int)
        # self.grid = sparse.coo_matrix(([], ([], [])), shape=(self.N_ROWS, self.N_COLS)).astype(np.int8)
        self.neighbors = SparseDict(lambda: SparseDict(0))  # access like neighbors[(i,j)][(k,l)]
        self.legal_actions = list(range(self.N_COLS))
        self.is_terminal_state = False
        self.cur_player_index = 0
        self.next_rows = ((self.N_ROWS-1) * np.ones(self.N_COLS)).astype(int)
        self.rewards = np.zeros(2)

    def __eq__(self, other):
        return (self.grid == other.grid).all()


class Connect4(SequentialGame):
    @classmethod
    def is_terminal_state(cls, state: Connect4State):
        return state.is_terminal_state

    @classmethod
    def get_cur_player_index(cls, state: Connect4State) -> int:
        return state.cur_player_index

    @classmethod
    def get_next_state_and_rewards(cls, state: Connect4State, action):
        next_state = copy.deepcopy(state)
        marker = 1 - 2*state.cur_player_index
        i = next_state.next_rows[action]
        j = action
        next_state.grid[0, i, j] = marker

        next_state.next_rows[action] -= 1
        if next_state.next_rows[action] < 0:
            next_state.legal_actions.remove(action)

        for k in range(3):
            for l in range(3):
                if k == 1 and l == 1:
                    continue

                direction_x = l - 1
                direction_y = k - 1
                if (j + direction_x) < 0 or (j + direction_x) >= Connect4State.N_COLS:
                    continue

                if (i + direction_y) < 0 or (i + direction_y) >= Connect4State.N_ROWS:
                    continue

                if next_state.grid[0, (i+direction_y), (j+direction_x)] == marker:
                    N = 1 + next_state.neighbors[((i+direction_y), (j+direction_x))][(k, l)]
                    next_state.neighbors.getcreate((i, j))[(k, l)] = N

                    if N >= 3:
                        next_state.is_terminal_state = True
                        next_state.rewards[state.cur_player_index] = 1
                        next_state.rewards[(1-state.cur_player_index)] = -1

        for k in range(3):
            for l in range(3):
                if k == 1 and l == 1:
                    continue

                direction_x = l - 1
                direction_y = k - 1
                if (j + direction_x) < 0 or (j + direction_x) >= Connect4State.N_COLS:
                    continue

                if (i + direction_y) < 0 or (i + direction_y) >= Connect4State.N_ROWS:
                    continue

                if next_state.grid[0, (i+direction_y), (j+direction_x)] == marker:
                    N = next_state.neighbors[(i, j)][(k, l)]
                    if N == 0:
                        continue

                    O = next_state.neighbors[(i, j)][((2-k), (2-l))]
                    next_state.neighbors.getcreate(((i+N*direction_y), (j+N*direction_x)))[(2-k, 2-l)] = (N+O)

                    if (N+O) >= 3:
                        next_state.is_terminal_state = True
                        next_state.rewards[state.cur_player_index] = 1
                        next_state.rewards[(1-state.cur_player_index)] = -1

        if not next_state.is_terminal_state:
            if next_state.grid[0].abs().sum() == (Connect4State.N_COLS * Connect4State.N_ROWS):
                next_state.is_terminal_state = True

        next_state.cur_player_index = 1 - state.cur_player_index
        return next_state, next_state.rewards

    @classmethod
    def get_rewards(cls, state: Connect4State):
        return state.rewards

    @classmethod
    def get_all_actions(cls) -> List:
        return list(range(7))

    @classmethod
    def get_legal_actions(cls, state: Connect4State) -> List:
        return state.legal_actions

    @classmethod
    def get_n_players(cls):
        return 2

    @classmethod
    def states_equal(cls, state1: Connect4State, state2: Connect4State):
        return state1 == state2

    @classmethod
    def get_initial_state(cls):
        return Connect4State()
