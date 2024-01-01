from typing import List
import numpy as np
from .game import SequentialGame
import enum
import copy


def get_random_new_value():
    if np.random.rand() < 0.9:
        return 2
    else:
        return 4


class TwentyFortyEightState:
    def __init__(self):
        self.grid = np.zeros((4, 4)).astype(int)
        random_position = tuple(np.random.randint(4, size=2))

        random_position2 = tuple(np.random.randint(4, size=2))
        while random_position2 == random_position:
            random_position2 = tuple(np.random.randint(4, size=2))

        # For each random position, put a value in
        for pos in [random_position, random_position2]:
            self.grid[pos] = get_random_new_value()

        self.reward = np.array([0])


class TwentyFortyEight(SequentialGame):
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
        return len(cls.get_legal_actions(state)) == 0

    @classmethod
    def get_cur_player_index(cls, state) -> int:
        return 0

    @classmethod
    def get_next_state_and_rewards(cls, state, action):
        state = copy.deepcopy(state)
        state.reward = 0

        state_to_shift = state.grid
        if action == cls.Moves.RIGHT:
            state_to_shift = np.flip(state.grid, 1)
        elif action == cls.Moves.UP:
            state_to_shift = state.grid.T
        elif action == cls.Moves.DOWN:
            state_to_shift = np.flip(state.grid.T, 1)

        for row in state_to_shift:
            state.reward += cls.shift_row_left(row)

        # Add in a random 2:
        empty_slots = list(zip(*np.where(state.grid == 0)))
        random_empty_slot = empty_slots[np.random.randint(len(empty_slots))]
        state.grid[random_empty_slot] = get_random_new_value()

        return state, [state.reward]

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
            if np.sum(row[pos:]) > 0: # Shift things over if there are non-zeros to shift over
                while row[pos] == 0:
                    row[pos:] = np.append(row[(pos+1):], 0)

        for pos in range(len(row) - 1):
            if row[pos] == row[pos+1]:
                row[pos] *= 2
                row[(pos+1):] = np.append(row[(pos+2):], 0)
                merge_values += row[pos]

        return merge_values

    @classmethod
    def get_initial_state(cls):
        return TwentyFortyEightState()

    @classmethod
    def get_all_actions(cls) -> List:
        return list(cls.Moves)

    @classmethod
    def get_legal_actions(cls, state) -> List:
        actions = []

        grid = state.grid

        if any(cls.row_can_move_left(row) for row in grid):
            actions.append(cls.Moves.LEFT)

        if any(cls.row_can_move_right(row) for row in grid):
            actions.append(cls.Moves.RIGHT)

        if any(cls.col_can_move_up(col) for col in grid.T):
            actions.append(cls.Moves.UP)

        if any(cls.col_can_move_down(col) for col in grid.T):
            actions.append(cls.Moves.DOWN)

        return actions

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
