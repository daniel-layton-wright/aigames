from aigames import *
import numpy as np
import itertools
from typing import Union


class TicTacToe(SequentialGame):
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
        state = copy.deepcopy(state)

        player_index = cls.get_cur_player_index(state)
        player_marker = 1 if player_index == 0 else -1
        state[action] = player_marker

        rewards = np.zeros(2)
        if cls.is_terminal_state(state):
            rewards = cls.get_terminal_rewards(state)

        return state, rewards

    @classmethod
    def get_winner(cls, state) -> Union[int, None]:
        if any((mask * state).sum() == 3 for mask in cls.masks):
            return 0
        elif any((mask * state).sum() == -3 for mask in cls.masks):
            return 1
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
            '-----\n'.join(['{}|{}|{}\n'.format(*line_to_marks(self.state[i, :])) for i in
                            range(self.state.shape[0])])
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

        new_state = (self.state[0, :, :] - self.state[1, :, :]) * (1 - 2 * self.state[2, 0, 0])

        return (
            '-----\n'.join(['{}|{}|{}\n'.format(*line_to_marks(new_state[i, :])) for i in
                            range(new_state.shape[0])])
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
            '-----\n'.join(['{}|{}|{}\n'.format(*line_to_marks(self.state[i, :])) for i in
                            range(self.state.shape[0])])
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