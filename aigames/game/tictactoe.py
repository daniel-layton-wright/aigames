from aigames import *
import numpy as np
import itertools
from typing import Union
import torch
from functools import lru_cache


class TicTacToe(SequentialGame):
    WIN_REWARD = 1
    TIE_REWARD = 0
    LOSE_REWARD = -1

    @classmethod
    def get_n_players(cls):
        return 2

    @classmethod
    def get_initial_state(cls):
        return np.zeros((3, 3))

    @classmethod
    def get_all_actions(cls):
        return list(itertools.product(range(3), range(3)))

    @classmethod
    def is_terminal_state(cls, state):
        #return any(abs((mask * state).sum()) == 3 for mask in cls.masks) or abs(state).sum() == 9
        # Brute force
        return (
            abs(state).sum() == 9 or
            (state[0][0] == 1 and state[0][1] == 1 and state[0][2] == 1) or
            (state[1][0] == 1 and state[1][1] == 1 and state[1][2] == 1) or
            (state[2][0] == 1 and state[2][1] == 1 and state[2][2] == 1) or
            (state[0][0] == 1 and state[1][0] == 1 and state[2][0] == 1) or
            (state[0][1] == 1 and state[1][1] == 1 and state[2][1] == 1) or
            (state[0][2] == 1 and state[1][2] == 1 and state[2][2] == 1) or
            (state[0][0] == 1 and state[1][1] == 1 and state[2][2] == 1) or
            (state[0][2] == 1 and state[1][1] == 1 and state[2][0] == 1) or
            (state[0][0] == -1 and state[0][1] == -1 and state[0][2] == -1) or
            (state[1][0] == -1 and state[1][1] == -1 and state[1][2] == -1) or
            (state[2][0] == -1 and state[2][1] == -1 and state[2][2] == -1) or
            (state[0][0] == -1 and state[1][0] == -1 and state[2][0] == -1) or
            (state[0][1] == -1 and state[1][1] == -1 and state[2][1] == -1) or
            (state[0][2] == -1 and state[1][2] == -1 and state[2][2] == -1) or
            (state[0][0] == -1 and state[1][1] == -1 and state[2][2] == -1) or
            (state[0][2] == -1 and state[1][1] == -1 and state[2][0] == -1)
        )

    @classmethod
    def get_cur_player_index(cls, state) -> int:
        return int(abs(state).sum()) % 2

    @classmethod
    def get_legal_actions(cls, state) -> List:
        return sorted(list(set(zip(*np.where(state == 0)))))

    @classmethod
    def get_next_state_and_rewards(cls, state, action):
        state = copy.deepcopy(state)

        player_index = cls.get_cur_player_index(state)
        player_marker = 1 if player_index == 0 else -1
        state[action] = player_marker

        rewards = cls.get_rewards(state)
        return state, rewards

    @classmethod
    def get_rewards(cls, state):
        if cls.is_terminal_state(state):
            return cls.get_terminal_rewards(state)
        else:
            return np.zeros(2)

    @classmethod
    def get_winner(cls, state) -> Union[int, None]:
        if (
            (state[0][0] == 1 and state[0][1] == 1 and state[0][2] == 1) or
            (state[1][0] == 1 and state[1][1] == 1 and state[1][2] == 1) or
            (state[2][0] == 1 and state[2][1] == 1 and state[2][2] == 1) or
            (state[0][0] == 1 and state[1][0] == 1 and state[2][0] == 1) or
            (state[0][1] == 1 and state[1][1] == 1 and state[2][1] == 1) or
            (state[0][2] == 1 and state[1][2] == 1 and state[2][2] == 1) or
            (state[0][0] == 1 and state[1][1] == 1 and state[2][2] == 1) or
            (state[0][2] == 1 and state[1][1] == 1 and state[2][0] == 1)
        ):
            return 0
        elif (
            (state[0][0] == -1 and state[0][1] == -1 and state[0][2] == -1) or
            (state[1][0] == -1 and state[1][1] == -1 and state[1][2] == -1) or
            (state[2][0] == -1 and state[2][1] == -1 and state[2][2] == -1) or
            (state[0][0] == -1 and state[1][0] == -1 and state[2][0] == -1) or
            (state[0][1] == -1 and state[1][1] == -1 and state[2][1] == -1) or
            (state[0][2] == -1 and state[1][2] == -1 and state[2][2] == -1) or
            (state[0][0] == -1 and state[1][1] == -1 and state[2][2] == -1) or
            (state[0][2] == -1 and state[1][1] == -1 and state[2][0] == -1)
        ):
            return 1
        else:
            return None

    @classmethod
    def get_terminal_rewards(cls, state):
        rewards = np.ones(2) * cls.TIE_REWARD
        winner = cls.get_winner(state)
        if winner is not None:
            loser = 1 - winner
            rewards[winner] = cls.WIN_REWARD
            rewards[loser] = cls.LOSE_REWARD

        return rewards

    def __str__(self):
        return board_array_to_string(self.state)

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

    @classmethod
    def states_equal(cls, state1, state2):
        return np.all(state1 == state2)


def custom_reward_tictactoe(win_reward, tie_reward, lose_reward):
    class CustomRewardTicTacToe(TicTacToe):
        WIN_REWARD = win_reward
        TIE_REWARD = tie_reward
        LOSE_REWARD = lose_reward

    return CustomRewardTicTacToe


class TicTacToe2(SequentialGame):
    @classmethod
    def get_n_players(cls):
        return 2

    def get_initial_state(self):
        return np.zeros((3, 3, 3)).astype(int)

    @classmethod
    def get_all_actions(cls):
        return list(itertools.product(range(3), range(3)))

    @classmethod
    def is_terminal_state(cls, state):
        return any(abs((mask * state[i, :, :]).sum()) == 3 for mask in cls.masks for i in range(2)) \
                   or abs(state[:2, :, :]).sum() == 9

    @classmethod
    def get_cur_player_index(cls, state) -> int:
        return state[2, 0, 0]

    @classmethod
    def get_legal_actions(cls, state) -> List:
        return sorted(list(set(zip(*np.where(state[:2, :, :].sum(axis=0) == 0)))))

    @classmethod
    def get_next_state_and_rewards(cls, state, action):
        state = copy.deepcopy(state)

        state[(0, *action)] = 1
        state[[0, 1], :, :] = state[[1, 0], :, :]
        state[2, : , :] = 1 - state[2, 0, 0]

        rewards = np.zeros(2)
        if cls.is_terminal_state(state):
            rewards = cls.get_terminal_rewards(state)

        return state, rewards

    @classmethod
    def get_winner(cls, state) -> int:
        return state[2, 0, 0] if any((mask * state[0, :, :]).sum() == 3 for mask in cls.masks) else (1 - state[2, 0, 0])

    @classmethod
    def get_terminal_rewards(cls, state):
        rewards = np.zeros(2)
        winner = cls.get_winner(state)
        if winner is not None:
            loser = 1 - winner
            rewards[winner] = 1
            rewards[loser] = -1

        return rewards

    def __str__(self):
        new_state = (self.state[0, :, :] - self.state[1, :, :]) * (1 - 2 * self.state[2, 0, 0])
        return board_array_to_string(new_state)

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


class TicTacToe3(SequentialGame):
    @classmethod
    def get_n_players(cls):
        return 2

    def get_initial_state(self):
        return np.zeros((3, 3))

    @classmethod
    def get_all_actions(cls):
        return list(itertools.product(range(3), range(3)))

    @classmethod
    def is_terminal_state(cls, state):
        return any(abs((mask * state).sum()) == 3 for mask in cls.masks) or abs(state).sum() == 9

    @classmethod
    def get_cur_player_index(cls, state) -> int:
        return int(abs(state).sum()) % 2

    @classmethod
    def get_legal_actions(cls, state) -> List:
        return sorted(list(set(zip(*np.where(state == 0)))))

    @classmethod
    def get_next_state_and_rewards(cls, state, action):
        state = -1 * copy.deepcopy(state)
        state[action] = -1

        rewards = np.zeros(2)
        if cls.is_terminal_state(state):
            rewards = cls.get_terminal_rewards(state)

        return state, rewards

    @classmethod
    def get_winner(cls, state) -> Union[int, None]:
        if any((mask * state).sum() == -3 for mask in cls.masks):
            return 1 - int(abs(state).sum()) % 2
        else:
            return None

    @classmethod
    def get_terminal_rewards(cls, state):
        rewards = np.zeros(2)
        winner = cls.get_winner(state)
        if winner is not None:
            loser = 1 - winner
            rewards[winner] = 1
            rewards[loser] = -1

        return rewards

    def __str__(self):
        return board_array_to_string(self.state)

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


class FastTicTacToeState:
    def __init__(self, np_state):
        self.tensor_state = torch.FloatTensor(np_state).unsqueeze(0)
        self.legal_actions = TicTacToe.get_legal_actions(np_state)
        self.cur_player_index = TicTacToe.get_cur_player_index(np_state)
        self.is_terminal_state = TicTacToe.is_terminal_state(np_state)
        self.rewards = np.zeros(2)

    def __eq__(self, other):
        return (self.tensor_state == other.tensor_state).all()

    def hash(self):
        return tuple(self.tensor_state[0].numpy().flatten())

    def __hash__(self):
        return self.hash().__hash__()

    def __str__(self):
        return board_array_to_string(self.tensor_state[0].numpy())


class FastTicTacToe(SequentialGame):
    @classmethod
    def get_initial_state(cls):
        initial_np_state = TicTacToe.get_initial_state()
        return FastTicTacToeState(initial_np_state)

    @classmethod
    def get_legal_actions(cls, state: FastTicTacToeState) -> List:
        return state.legal_actions

    @classmethod
    def get_all_actions(cls) -> List:
        return TicTacToe.get_all_actions()

    @classmethod
    def get_cur_player_index(cls, state: FastTicTacToeState) -> int:
        return state.cur_player_index

    @classmethod
    def get_n_players(cls):
        return TicTacToe.get_n_players()

    @classmethod
    def is_terminal_state(cls, state: FastTicTacToeState):
        return state.is_terminal_state

    @classmethod
    @lru_cache(maxsize=1000)
    def get_next_state_and_rewards(cls, state, action):
        next_state = cls.get_initial_state()
        m = (1-2*state.cur_player_index)
        c = state.cur_player_index
        next_state.tensor_state = torch.clone(state.tensor_state)
        next_state.tensor_state[(0, *action)] = m
        next_state.cur_player_index = 1 - c
        next_state.legal_actions = state.legal_actions[:]
        next_state.legal_actions.remove(action)

        s = next_state.tensor_state[0]
        if s.abs().sum() == 9:
            next_state.is_terminal_state = True
        elif (
               (s[action[0], 0] == m and s[action[0], 1] == m and s[action[0], 2] == m)
            or (s[0, action[1]] == m and s[1, action[1]] == m and s[2, action[1]] == m)
            or (action[0] == action[1] and s[0, 0] == m and s[1, 1] == m and s[2, 2] == m)
            or (action[0] + action[1] == 2 and s[0,2] == m and s[1, 1] == m and s[2, 0] == m)
            ):
                next_state.is_terminal_state = True
                next_state.rewards[c] = 1
                next_state.rewards[(1-c)] = -1

        return next_state, next_state.rewards

    @classmethod
    def get_rewards(cls, state: FastTicTacToeState):
        return state.rewards

    @classmethod
    def states_equal(cls, state1, state2):
        return state1 == state2

    @classmethod
    def get_terminal_rewards(cls, state: FastTicTacToeState):
        state = state.tensor_state[0].numpy()
        rewards = np.zeros(2)
        winner = TicTacToe.get_winner(state)
        if winner is not None:
            loser = 1 - winner
            rewards[winner] = 1
            rewards[loser] = -1

        return rewards

    def __str__(self):
        out = board_array_to_string(self.state.tensor_state[0].numpy())
        out += f'\n{self.state.rewards}'
        return out


def board_array_to_string(arr):
    def line_to_marks(line):
        out = []
        for i in line:
            if i == 1:
                out.append('x')
            elif i == -1:
                out.append('o')
            else:
                out.append(' ')

        return out

    return (
        '-----\n'.join(['{}|{}|{}\n'.format(*line_to_marks(arr[i, :])) for i in
                        range(arr.shape[0])])
    )


def board_string_to_array(s):
    """
    Convert a string of the format xxo|o-x|oox to a numpy array
    """
    def char_to_int(c):
        if c == 'x':
            return 1
        elif c == 'o':
            return -1
        else:
            return 0

    return np.array([[char_to_int(c) for c in line] for line in s.split('|')])
