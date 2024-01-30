from typing import List, Tuple
import numpy as np
import enum
import torch
from .G2048 import G2048
from aigames.game.game_multi import GameMulti


def get_random_new_value():
    if np.random.rand() < 0.9:
        return 1
    else:
        return 2


def get_next_states_core(states: torch.Tensor, actions: torch.Tensor, move_left_map: torch.Tensor,
                         move_right_map: torch.Tensor, reward_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    lookup_tensor = torch.FloatTensor([16 ** 3, 16 ** 2, 16, 1], device=states.device).unsqueeze(1)
    new_states = torch.zeros_like(states)
    rewards = torch.zeros((states.shape[0], 1), dtype=torch.float32, device=states.device)
    left_mask = (actions == 0)
    right_mask = (actions == 1)
    up_mask = (actions == 2)
    down_mask = (actions == 3)

    left_lookup_idx = states[left_mask, :, :].matmul(lookup_tensor).flatten().to(int)
    new_states[left_mask, :, :] = (move_left_map[left_lookup_idx, :].reshape((left_mask.sum(), 4, 4)))
    rewards[left_mask, :] = reward_map[left_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    right_lookup_idx = states[right_mask, :, :].matmul(lookup_tensor).flatten().to(int)
    new_states[right_mask, :, :] = (move_right_map[right_lookup_idx, :].reshape((right_mask.sum(), 4, 4)))
    rewards[right_mask, :] = reward_map[right_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    up_lookup_idx = lookup_tensor.T.matmul(states[up_mask, :, :]).flatten().to(int)
    new_states[up_mask, :, :] = (move_left_map[up_lookup_idx, :].reshape((up_mask.sum(), 4, 4))).transpose(1, 2)
    rewards[up_mask, :] = reward_map[up_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    down_lookup_idx = lookup_tensor.T.matmul(states[down_mask, :, :]).flatten().to(int)
    new_states[down_mask, :, :] = (
        move_right_map[down_lookup_idx, :].reshape((down_mask.sum(), 4, 4))).transpose(1, 2)
    rewards[down_mask, :] = reward_map[down_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    # This can be used to give a reward for keeping highest tile in bottom left. 512 is the reward value
    # rewards += ((new_states[:, 3, 0] == new_states.amax(dim=(1, 2))) * 512).unsqueeze(1)

    return new_states, rewards


def get_next_states_full(states: torch.Tensor, actions: torch.Tensor, move_left_map: torch.Tensor,
                         move_right_map: torch.Tensor, reward_map: torch.Tensor, legal_move_mask: torch.Tensor)\
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    new_states, rewards = get_next_states_core(states, actions, move_left_map, move_right_map, reward_map)

    # Set is terminal
    is_terminal = is_terminal_core(new_states, legal_move_mask)

    env_is_next = torch.ones((states.shape[0],), dtype=torch.bool, device=states.device) & ~is_terminal

    return new_states, rewards, env_is_next, is_terminal


def is_terminal_core(states: torch.Tensor, legal_move_mask: torch.Tensor) -> torch.Tensor:
    legal_action_masks = get_legal_action_masks_core(states, legal_move_mask)
    return legal_action_masks.sum(dim=1) == 0


def get_legal_action_masks_core(states, legal_move_mask):
    lookup_tensor = torch.FloatTensor([16 ** 3, 16 ** 2, 16, 1], device=states.device).unsqueeze(1)
    ind = states.matmul(lookup_tensor).flatten().to(int)
    mask = legal_move_mask[ind, :].reshape(states.shape[0], 4, 2).any(dim=1)

    ind = lookup_tensor.T.matmul(states).flatten().to(int)
    return torch.cat([mask, legal_move_mask[ind, :].reshape(states.shape[0], 4, 2).any(dim=1)], dim=1)


def get_next_states_from_env_core(states: torch.Tensor, legal_move_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ones = torch.eye(16, dtype=states.dtype).view(16, 4, 4)
    twos = torch.eye(16, dtype=states.dtype).view(16, 4, 4) * 2
    base_progressions = torch.concat([ones, twos], dim=0).to(states.device)
    base_probabilities = torch.concat([torch.full((16,), 0.9), torch.full((16,), 0.1)], dim=0).to(states.device)
    valid_progressions = torch.logical_not(torch.any((states.unsqueeze(1) * base_progressions).view(-1, 32, 16), dim=2))
    progressions = (states.unsqueeze(1) + base_progressions) * valid_progressions.view(states.shape[0], 32, 1, 1)
    probs = base_probabilities * valid_progressions

    return progressions, probs


class G2048Multi(GameMulti):

    lookup_tensor = torch.FloatTensor([16 ** 3, 16 ** 2, 16, 1]).unsqueeze(1)

    get_next_states_jit = torch.jit.trace(
        get_next_states_full,
        example_inputs=(torch.randint(0, 2, (2, 4, 4), dtype=torch.float32), torch.LongTensor([0, 0]),
                        torch.randint(0, 2, (16**4, 4), dtype=torch.float32),
                        torch.randint(0, 2, (16**4, 4), dtype=torch.float32),
                        torch.randint(0, 2, (16**4, 1), dtype=torch.float32),
                        torch.randint(0, 2, (16**4, 2), dtype=torch.float32))
    )

    get_legal_action_masks_jit = torch.jit.trace(
        get_legal_action_masks_core,
        example_inputs=(torch.randint(0, 2, (2, 4, 4), dtype=torch.float32), torch.randint(0, 2, (16**4, 2), dtype=torch.bool))
    )

    is_terminal_jit = torch.jit.trace(
        is_terminal_core,
        example_inputs=(torch.randint(0, 2, (2, 4, 4), dtype=torch.float32), torch.randint(0, 2, (16**4, 2), dtype=torch.bool))
    )

    get_next_states_from_env_jit = torch.jit.trace(
        get_next_states_from_env_core,
        example_inputs=(torch.randint(0, 2, (2, 4, 4), dtype=torch.float32), torch.randint(0, 2, (16**4, 2), dtype=torch.bool))
    )

    def __init__(self, n_parallel_games, player, listeners):
        self.device = 'cpu'
        super().__init__(n_parallel_games, player, listeners)

    def set_device(self, device):
        self.device = device

        MOVE_LEFT_MAP.to(self.device)
        MOVE_RIGHT_MAP.to(self.device)
        REWARD_MAP.to(self.device)
        LEGAL_MOVE_MASK.to(self.device)

    def get_n_players(self):
        return 1

    def get_n_actions(self):
        return 4

    def get_n_stochastic_actions(self):
        return 32

    def get_state_shape(self) -> Tuple[int, ...]:
        return 4, 4

    class Moves(enum.Enum):
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3

    def is_terminal(self, states: torch.FloatTensor):
        # If any zeros, not terminal, else if no legal actions, terminal
        # The shape of states is N x state shape so N x 4 x 4
        # We need to return bools of size (N,)
        return self.is_terminal_jit(states, LEGAL_MOVE_MASK)

    def get_cur_player_index(self, states) -> torch.Tensor:
        return torch.zeros((states.shape[0],), dtype=torch.long)

    def get_next_states(self, states: torch.Tensor, actions: torch.Tensor):
        """
        :param states: A tensor of size (N, 4, 4) representing the states
        :param actions: A tensor of size (N,) representing the actions
        """
        return self.get_next_states_jit(states, actions, MOVE_LEFT_MAP, MOVE_RIGHT_MAP, REWARD_MAP, LEGAL_MOVE_MASK)

    def get_next_states_from_env(self, states: torch.Tensor):
        """
        For each state in the states tensor, we need to add a random 1 or 2 into a slot that is zero

        :param states: A tensor of size (N, 4, 4) representing the states
        """
        next_states, probs = self.get_next_states_from_env_jit(states, LEGAL_MOVE_MASK)

        idx = torch.multinomial(probs, num_samples=1).flatten()
        states = next_states[torch.arange(states.shape[0], device=states.device), idx, :, :]
        is_terminal = self.is_terminal(states)

        return states, idx, is_terminal

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
                merge_values += row[pos]

        return merge_values

    def get_initial_states(self, n_games):
        states = torch.zeros((n_games, 4, 4), dtype=torch.float32, device=self.device)

        # Add in two random values for each state
        states, _, _ = self.get_next_states_from_env(states)
        states, _, _ = self.get_next_states_from_env(states)

        return states

    def get_legal_action_masks(self, states: torch.FloatTensor):
        """
        :param states: A tensor of size (N, 4, 4) representing the states
        """
        return self.get_legal_action_masks_jit(states, LEGAL_MOVE_MASK)

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
    def row_can_move_right(cls, row):
        return cls.row_can_move_left(np.flip(row, 0))

    @classmethod
    def col_can_move_up(cls, col):
        return cls.row_can_move_left(col)

    @classmethod
    def col_can_move_down(cls, col):
        return cls.row_can_move_right(col)

    def __str__(self):
        def line_to_str(line, padding):
            out = []
            for i in line:
                out.append(str(i.item()).ljust(padding))

            return out

        grid = self.states.reshape(-1, 4)
        padding = len(str(grid.max()))
        line_length = padding*4 + 3
        return (
            ('-' * line_length + '\n').join(['{}|{}|{}|{}\n'.format(*line_to_str(grid[i, :], padding)) for i in range(grid.shape[0])])
        )


def get_move_left_map_and_rewards():
    move_left_map = torch.zeros((16**4, 4), dtype=torch.float)
    reward_map = torch.zeros((16**4, 1), dtype=torch.float)
    lookup_tensor = torch.FloatTensor([16**3, 16**2, 16, 1])
    for i in range(16):
        for j in range(16):
            for k in range(16):
                for l in range(16):
                    row = torch.FloatTensor([i, j, k, l])
                    ind = int(lookup_tensor.dot(row).item())
                    reward = G2048.shift_row_left(row)
                    move_left_map[ind, :] = row
                    reward_map[ind] = reward

    return move_left_map, reward_map


def get_move_right_map():
    move_right_map = torch.zeros((16**4, 4), dtype=torch.float)
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
                    move_right_map[ind, :] = row

    return move_right_map


def get_legal_move_masks():
    """
    Fills in the LEGAL_MOVE_MASK array for a row. The format is [LEFT, RIGHT] (use transposes to get up down)
    """
    legal_move_mask = torch.zeros((16**4, 2), dtype=torch.bool)
    lookup_tensor = torch.FloatTensor([16**3, 16**2, 16, 1])
    for i in range(16):
        for j in range(16):
            for k in range(16):
                for l in range(16):
                    row = torch.FloatTensor([i, j, k, l])
                    ind = int(lookup_tensor.dot(row).item())
                    legal_move_mask[ind, 0] = G2048.row_can_move_left(row)
                    legal_move_mask[ind, 1] = G2048.row_can_move_left(torch.flip(row, [0]))

    return legal_move_mask


MOVE_LEFT_MAP, REWARD_MAP = get_move_left_map_and_rewards()
MOVE_RIGHT_MAP = get_move_right_map()
LEGAL_MOVE_MASK = get_legal_move_masks()
