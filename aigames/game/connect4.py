from functools import lru_cache
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

    def clone(self):
        out = SparseDict(self.default_value)
        out.data = copy.deepcopy(self.data)
        return out


def copy_neighbors(neighbors: SparseDict):
    """
    This is faster than deepcopy or clone (but we leverage knowing the structure of the variable, so it isn't all-purpose)

    :param neighbors: SparseDict of SparseDicts
    """
    out = SparseDict(neighbors.default_value)
    for key, value in neighbors.data.items():
        out[key] = SparseDict(value.default_value)
        out[key].data = value.data.copy()

    return out


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

    def __hash__(self):
        return tuple(self.grid[0].numpy().flatten()).__hash__()


class Connect4(SequentialGame):
    @classmethod
    def is_terminal_state(cls, state: Connect4State):
        return state.is_terminal_state

    @classmethod
    def get_cur_player_index(cls, state: Connect4State) -> int:
        return state.cur_player_index

    @classmethod
    @lru_cache(maxsize=32000)
    def get_next_state_and_rewards(cls, state: Connect4State, action):
        # This is faster than deepcopy
        next_state = Connect4State()
        next_state.neighbors = copy_neighbors(state.neighbors)
        next_state.grid = state.grid.detach().clone()
        next_state.legal_actions = state.legal_actions[:]
        next_state.next_rows = state.next_rows.copy()

        marker = 1 - 2*state.cur_player_index  # either 1 or -1
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

    def __str__(self):
        # Print grid as x's and o's
        out = ""
        for i in range(Connect4State.N_ROWS):
            out += '|'
            for j in range(Connect4State.N_COLS):
                if self.state.grid[0, i, j] == 1:
                    out += "x|"
                elif self.state.grid[0, i, j] == -1:
                    out += "o|"
                else:
                    out += " |"
            out += "\n"

        out += '‾' * (Connect4State.N_COLS * 2 + 1)

        return out


class Connect4BitState:
    """
    The state is represented as two bit strings (game_grid and player0_grid) of length 49. The positions are laid out
    as follows:

    - -  -  -  -  -  -
    6 13 20 27 34 41 48
    5 12 19 26 33 40 47
    4 11 18 25 32 39 46
    3 10 17 24 31 38 45
    2 9  16 23 30 37 44
    1 8  15 22 29 36 43

    Note that in positions 7, 14, ..., 49, we always have a zero. This is for convenience when checking for four in a
    row vertically (to make sure we don't wrap around to count four in a row mistakenly)

    We also store the tensor_state for convenience because this is needed by the network and rather than recompute it
    every network evaluation, we can just update it progressively in this state. But it is not used for the game logic.

    """
    N_ROWS = 6
    N_COLS = 7

    def __init__(self):
        self.game_grid = 0  # this has a 1 where either player has played
        self.player0_grid = 0  # this has a 1 where player 0 has played
        self.cur_player_index = 0
        self.next_rows = np.zeros(self.N_COLS).astype(int)
        self.legal_actions = list(range(self.N_COLS))
        self.tensor_state = torch.zeros(2, self.N_ROWS+1, self.N_COLS)

    def clone(self):
        new_state = Connect4BitState()
        new_state.game_grid = self.game_grid
        new_state.player0_grid = self.player0_grid
        new_state.cur_player_index = self.cur_player_index
        new_state.next_rows = self.next_rows.copy()
        new_state.legal_actions = self.legal_actions[:]
        new_state.tensor_state = self.tensor_state.clone()
        return new_state

    @property
    def player1_grid(self):
        return self.game_grid ^ self.player0_grid  # you can tell where player1 played where there is a 1 in game_grid but not in player0_grid

    @property
    def is_terminal_state(self):
        return (self.is_winning_state(self.player0_grid) or self.is_winning_state(self.player1_grid) or
                self.next_rows.sum() == self.N_COLS * self.N_ROWS)

    @staticmethod
    def is_winning_state(grid):
        # Check horizontal 4 in a row
        horizontal_neighbor = grid & (grid >> 7)
        if (horizontal_neighbor & (horizontal_neighbor >> 14)) > 0:
            # If a square has a horizontal neighbor and the square two over also has a horizontal neighbor,
            #   then there is 4 in a row
            return True

        # Check vertical 4 in a row
        vertical_neighbor = grid & (grid >> 1)
        if (vertical_neighbor & (vertical_neighbor >> 2)) > 0:
            return True

        # Check diagonal 4 in a row
        diagonal_neighbor = grid & (grid >> 8)
        if (diagonal_neighbor & (diagonal_neighbor >> 16)) > 0:
            return True

        # Check other diagonal 4 in a row
        other_diagonal_neighbor = grid & (grid >> 6)
        if (other_diagonal_neighbor & (other_diagonal_neighbor >> 12)) > 0:
            return True

        # If no 4 in a row found, return False
        return False

    @property
    def rewards(self):
        # If not a terminal state, return 0s
        if self.is_winning_state(self.player0_grid):
            return np.array([1, -1])
        elif self.is_winning_state(self.player1_grid):
            return np.array([-1, 1])
        else:
            return np.array([0, 0])

    def __eq__(self, other):
        return self.game_grid == other.game_grid and self.player0_grid == other.player0_grid

    def __hash__(self):
        return hash((self.game_grid, self.player0_grid))

    def __str__(self):
        out = ""
        for row in range(self.N_ROWS-1, -1, -1):
            out += "|"
            for column in range(0, 7):
                if self.player0_grid & (1 << (self.N_COLS * column + row)):
                    out += "x"
                elif self.player1_grid & (1 << (self.N_COLS * column + row)):
                    out += "o"
                else:
                    out += " "

                out += "|"

            out += "\n"

        return out


class Connect4V2(SequentialGame):
    ALL_ACTIONS = list(range(Connect4BitState.N_COLS))

    @classmethod
    @lru_cache(maxsize=100000)
    def get_next_state_and_rewards(cls, state: Connect4BitState, action):
        """

        :param state: the input state that action is being taken on (should not be changed by method)
        :param action: the column to play in
        """
        new_state = state.clone()
        new_state.cur_player_index = 1 - state.cur_player_index

        # Add a 1 to the game_grid in the appropriate location
        new_state.game_grid |= (1 << (Connect4BitState.N_COLS * action + new_state.next_rows[action]))

        # If this is player 0, also add a 1 to the player0_grid
        if state.cur_player_index == 0:
            new_state.player0_grid |= (1 << (Connect4BitState.N_COLS * action + new_state.next_rows[action]))

        # Update the tensor state
        new_state.tensor_state[0, Connect4BitState.N_ROWS - new_state.next_rows[action], action] = 1

        # Flip tensor state so current player is dim 0
        new_state.tensor_state = new_state.tensor_state.flip(0)

        # Update the next rows
        new_state.next_rows[action] += 1

        # Update legal actions, if necessary
        if new_state.next_rows[action] == Connect4BitState.N_ROWS:
            new_state.legal_actions.remove(action)

        return new_state, new_state.rewards

    @classmethod
    def is_terminal_state(cls, state):
        return state.is_terminal_state

    @classmethod
    def get_rewards(cls, state):
        return state.rewards

    @classmethod
    def get_all_actions(cls) -> List:
        return cls.ALL_ACTIONS

    @classmethod
    def get_legal_actions(cls, state: Connect4BitState) -> List:
        return state.legal_actions

    @classmethod
    def get_cur_player_index(cls, state: Connect4BitState) -> int:
        return state.cur_player_index

    @classmethod
    def get_n_players(cls):
        return 2

    @classmethod
    def states_equal(cls, state1: Connect4BitState, state2: Connect4BitState):
        return state1 == state2

    @classmethod
    def get_initial_state(cls):
        return Connect4BitState()

    def __str__(self):
        return str(self.state)
