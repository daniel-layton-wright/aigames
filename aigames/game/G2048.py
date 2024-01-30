from typing import List
import numpy as np
from .game import SequentialGame
import enum
import copy
import torch


def get_random_new_value():
    if np.random.rand() < 0.9:
        return 1
    else:
        return 2


class G2048State:
    def __init__(self):
        self.grid = torch.zeros((4, 4), dtype=torch.float)
        self.is_env_turn = True
        self.reward = torch.FloatTensor([0])


class G2048(SequentialGame):

    lookup_tensor = torch.FloatTensor([16 ** 3, 16 ** 2, 16, 1]).unsqueeze(1)

    @classmethod
    def get_n_players(cls):
        return 1

    class Moves(enum.Enum):
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3

    def __str__(self):
        def line_to_str(line, padding):
            out = []
            for i in line:
                out.append(str(i).ljust(padding))

            return out

        grid = self.state.grid
        padding = len(str(grid.max()))
        line_length = padding*4 + 3
        return (
            ('-' * line_length + '\n').join(['{}|{}|{}|{}\n'.format(*line_to_str(grid[i, :], padding)) for i in range(grid.shape[0])])
        )

    @classmethod
    def is_terminal_state(cls, state):
        # If any zeros, not terminal, else if no legal actions, terminal
        return (state.grid == 0).sum() == 0 and len(cls.get_legal_actions(state)) == 0

    @classmethod
    def get_cur_player_index(cls, state) -> int:
        return 0

    @classmethod
    def get_next_state_and_rewards(cls, state: G2048State, action):
        prev_max = torch.max(state.grid).item()
        state = copy.deepcopy(state)

        if action == cls.Moves.LEFT:
            state.grid = MOVE_LEFT_MAP[state.grid.matmul(cls.lookup_tensor).flatten().to(int), :, :].reshape((4, 4))
        elif action == cls.Moves.RIGHT:
            state.grid = MOVE_RIGHT_MAP[state.grid.matmul(cls.lookup_tensor).flatten().to(int), :, :].reshape((4, 4))
        elif action == cls.Moves.UP:
            state.grid = MOVE_LEFT_MAP[cls.lookup_tensor.T.matmul(state.grid).flatten().to(int), :, :].reshape((4, 4)).T
        elif action == cls.Moves.DOWN:
            state.grid = MOVE_RIGHT_MAP[cls.lookup_tensor.T.matmul(state.grid).flatten().to(int), :, :].reshape((4, 4)).T

        state.reward[:] = torch.max(state.grid).item() - prev_max

        # At the end, it's the env's turn to insert a random value
        state.is_env_turn = True

        return state, [state.reward]

    @classmethod
    def get_next_state_from_env(cls, state):
        state = copy.deepcopy(state)
        state.reward[:] = 0

        # If the state is empty we need to put in two values
        if state.grid.sum() == 0:
            # Add in a random 2 or 4
            empty_slots = list(zip(*np.where(state.grid == 0)))
            random_empty_slot = empty_slots[np.random.randint(len(empty_slots))]
            state.grid[random_empty_slot] = get_random_new_value()

            # Add in a random 2 or 4:
            empty_slots = list(zip(*np.where(state.grid == 0)))
            random_empty_slot = empty_slots[np.random.randint(len(empty_slots))]
            state.grid[random_empty_slot] = get_random_new_value()
        else:
            # Add in a random 2 or 4:
            empty_slots = list(zip(*np.where(state.grid == 0)))
            random_empty_slot = empty_slots[np.random.randint(len(empty_slots))]
            state.grid[random_empty_slot] = get_random_new_value()

        # Player's turn now
        state.is_env_turn = False
        return state

    @classmethod
    def get_rewards(cls, state):
        return [state.reward]

    @classmethod
    def shift_row_left(cls, row) -> int:
        """
        Shifts the row left (in place)

        :param row:
        :return: Point value of merges
        """
        merge_values = 0

        # First move everything over
        for pos in range(len(row)):
            if torch.sum(row[pos:]) > 0:  # Shift things over if there are non-zeros to shift over
                while row[pos] == 0:
                    row[pos:] = torch.cat([row[(pos+1):], torch.FloatTensor([0])])

        for pos in range(len(row) - 1):
            if row[pos] == 0:
                break

            if row[pos] == row[pos+1]:
                row[pos] += 1
                row[(pos+1):] = torch.cat([row[(pos+2):], torch.FloatTensor([0])])
                merge_values += 2**row[pos]

        return merge_values

    @classmethod
    def get_initial_state(cls):
        return G2048State()

    @classmethod
    def get_all_actions(cls) -> List:
        return list(cls.Moves)

    @classmethod
    def get_legal_actions_mask(cls, state):
        ind = state.grid.matmul(cls.lookup_tensor).flatten().to(int)
        mask = LEGAL_MOVE_MASK[ind, :, :].squeeze().any(dim=0)

        ind = cls.lookup_tensor.T.matmul(state.grid).flatten().to(int)
        return torch.cat([mask, LEGAL_MOVE_MASK[ind, :, :].squeeze().any(dim=0)])

    @classmethod
    def get_legal_actions(cls, state) -> List:
        mask = cls.get_legal_actions_mask(state)

        # Apply mask to the output of get_all_actions
        return np.array(cls.get_all_actions())[mask]

    @classmethod
    def row_can_move_left(cls, row):
        if any(i == j and i != 0 for i, j in zip(row[:-1], row[1:])):
            # A merge can be done
            return True

        first_zero = next((i for i, x in enumerate(row) if x == 0), None)
        last_non_zero = next((len(row) - i for i, x in enumerate(reversed(row)) if x != 0), None)
        if first_zero is not None and last_non_zero is not None and first_zero < last_non_zero:
            # The first zero is to the left of the last non-zero, so the row can shift over
            return True

        return False

    @classmethod
    def states_equal(cls, state1, state2):
        return np.array_equal(state1.grid, state2.grid) and state1.reward == state2.reward

    @classmethod
    def row_can_move_right(cls, row):
        return cls.row_can_move_left(np.flip(row, 0))

    @classmethod
    def col_can_move_up(cls, col):
        return cls.row_can_move_left(col)

    @classmethod
    def col_can_move_down(cls, col):
        return cls.row_can_move_right(col)

    @classmethod
    def is_env_turn(cls, state):
        return state.is_env_turn


MOVE_LEFT_MAP = torch.zeros((16**4, 1, 4), dtype=torch.float)
MOVE_RIGHT_MAP = torch.zeros((16**4, 1, 4), dtype=torch.float)
LEGAL_MOVE_MASK = torch.zeros((16**4, 1, 2), dtype=torch.bool)


def generate_move_left_map():
    lookup_tensor = torch.FloatTensor([16**3, 16**2, 16, 1])
    for i in range(16):
        for j in range(16):
            for k in range(16):
                for l in range(16):
                    row = torch.FloatTensor([i, j, k, l])
                    ind = int(lookup_tensor.dot(row).item())
                    G2048.shift_row_left(row)
                    MOVE_LEFT_MAP[ind, :, :] = row


def generate_move_right_map():
    lookup_tensor = torch.FloatTensor([16**3, 16**2, 16, 1])
    for i in range(16):
        for j in range(16):
            for k in range(16):
                for l in range(16):
                    row = torch.FloatTensor([i, j, k, l])
                    ind = int(lookup_tensor.dot(row).item())
                    row = torch.flip(row, [0])
                    G2048.shift_row_left(row)
                    row = torch.flip(row, [0])
                    MOVE_RIGHT_MAP[ind, :, :] = row


def generate_legal_move_masks():
    """
    Fills in the LEGAL_MOVE_MASK array for a row. The format is [LEFT, RIGHT] (use transposes to get up down)
    """
    lookup_tensor = torch.FloatTensor([16**3, 16**2, 16, 1])
    for i in range(16):
        for j in range(16):
            for k in range(16):
                for l in range(16):
                    row = torch.FloatTensor([i, j, k, l])
                    ind = int(lookup_tensor.dot(row).item())
                    LEGAL_MOVE_MASK[ind, :, 0] = G2048.row_can_move_left(row)
                    LEGAL_MOVE_MASK[ind, :, 1] = G2048.row_can_move_left(torch.flip(row, [0]))


generate_move_left_map()
generate_move_right_map()
generate_legal_move_masks()
