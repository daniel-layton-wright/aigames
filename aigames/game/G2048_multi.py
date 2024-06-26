from typing import Tuple, Final
import numpy as np
import enum
import torch
from aigames.game.game_multi import GameMulti
from aigames.utils.utils import cache


def get_next_states_core(states: torch.Tensor, actions: torch.Tensor, move_left_map: torch.Tensor,
                         move_right_map: torch.Tensor, reward_map: torch.Tensor, ind: torch.Tensor, indT: torch.Tensor)\
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    new_states = torch.zeros_like(states)
    rewards = torch.zeros((states.shape[0], 1), dtype=torch.float32, device=states.device)

    left_mask = (actions == 0)
    right_mask = (actions == 1)
    up_mask = (actions == 2)
    down_mask = (actions == 3)

    left_lookup_idx = ind[left_mask].flatten()
    new_states[left_mask, :, :] = (move_left_map[left_lookup_idx, :].reshape((left_mask.sum(), 4, 4)))
    rewards[left_mask, :] = reward_map[left_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True).to(torch.float32)

    right_lookup_idx = ind[right_mask].flatten()
    new_states[right_mask, :, :] = (move_right_map[right_lookup_idx, :].reshape((right_mask.sum(), 4, 4)))
    rewards[right_mask, :] = reward_map[right_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True).to(torch.float32)

    up_lookup_idx = indT[up_mask].flatten()
    new_states[up_mask, :, :] = (move_left_map[up_lookup_idx, :].reshape((up_mask.sum(), 4, 4))).transpose(1, 2)
    rewards[up_mask, :] = reward_map[up_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True).to(torch.float32)

    down_lookup_idx = indT[down_mask].flatten()
    new_states[down_mask, :, :] = (
        move_right_map[down_lookup_idx, :].reshape((down_mask.sum(), 4, 4))).transpose(1, 2)
    rewards[down_mask, :] = reward_map[down_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True).to(torch.float32)

    new_ind, new_indT = get_ind_core(new_states)

    return new_states, rewards, new_ind, new_indT


def get_ind_core(new_states):
    lookup_tensor: Final = torch.tensor([16 ** 3, 16 ** 2, 16, 1], dtype=torch.float32,
                                        device=new_states.device).unsqueeze(1)
    new_ind = new_states.to(torch.float32).matmul(lookup_tensor).to(torch.int32)
    new_indT = lookup_tensor.T.matmul(new_states.to(torch.float32)).to(torch.int32)
    return new_ind, new_indT


def get_ind(states):
    if hasattr(states, 'ind') and hasattr(states, 'indT'):
        return states.ind, states.indT
    else:
        return get_ind_core(states)


def get_next_states_full(states: torch.Tensor, actions: torch.Tensor, move_left_map: torch.Tensor,
                         move_right_map: torch.Tensor, reward_map: torch.Tensor, legal_move_mask: torch.Tensor,
                         ind: torch.Tensor, indT: torch.Tensor)\
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    new_states, rewards, new_ind, new_indT = get_next_states_core(states, actions, move_left_map, move_right_map,
                                                                  reward_map, ind, indT)

    # Set is terminal
    is_terminal = is_terminal_core(new_states, legal_move_mask, new_ind, new_indT)

    env_is_next = torch.ones((states.shape[0],), dtype=torch.bool, device=states.device) & ~is_terminal

    return new_states, rewards, env_is_next, is_terminal, new_ind, new_indT


def is_terminal_core(states: torch.Tensor, legal_move_mask: torch.Tensor, ind: torch.Tensor, indT: torch.Tensor)\
        -> torch.Tensor:
    legal_action_masks = get_legal_action_masks_core(states, legal_move_mask, ind, indT)
    return legal_action_masks.sum(dim=1) == 0


def get_legal_action_masks_core(states, legal_move_mask, ind, indT):
    mask = legal_move_mask[ind, :].reshape(states.shape[0], 4, 2).any(dim=1)
    return torch.cat([mask, legal_move_mask[indT, :].reshape(states.shape[0], 4, 2).any(dim=1)], dim=1)


def get_env_legal_action_masks_core(states: torch.Tensor):
    base_progressions = torch.concat([
        torch.eye(16, dtype=states.dtype).view(16, 4, 4),
        torch.eye(16, dtype=states.dtype).view(16, 4, 4) * 2
    ], dim=0).to(states.device)
    base_probabilities = torch.concat([
        torch.full((16,), 0.9),
        torch.full((16,), 0.1)
    ], dim=0).to(states.device)
    valid_progressions = torch.logical_not(torch.any((states.unsqueeze(1) * base_progressions).view(-1, 32, 16), dim=2))

    return valid_progressions


def get_next_states_from_env_core(states: torch.Tensor, random_values: torch.Tensor, legal_move_mask)\
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_progressions = torch.concat([
        torch.eye(16, dtype=states.dtype).view(16, 4, 4),
        torch.eye(16, dtype=states.dtype).view(16, 4, 4) * 2
    ], dim=0).to(states.device)
    base_probabilities = torch.concat([
        torch.full((16,), 0.9),
        torch.full((16,), 0.1)
    ], dim=0).to(states.device)
    valid_progressions = torch.logical_not(torch.any((states.unsqueeze(1) * base_progressions).view(-1, 32, 16), dim=2))
    probs = base_probabilities * valid_progressions
    probs /= probs.sum(dim=1, keepdim=True)
    idx = select_indices(probs, random_values)
    next_states = states + base_progressions[idx]

    next_ind, next_indT = get_ind_core(next_states)

    is_terminal = is_terminal_core(next_states, legal_move_mask, next_ind, next_indT)
    return next_states, idx, is_terminal, next_ind, next_indT


def get_next_states_from_env_action_idx(states: torch.Tensor, env_action_idx: torch.Tensor, legal_move_mask: torch.Tensor):
    mark = (env_action_idx // 16).to(torch.uint8) + 1
    idx = env_action_idx % 16
    idx1 = idx // 4
    idx2 = idx % 4
    next_states = states.clone()
    next_states[torch.arange(states.shape[0]), idx1, idx2] = mark

    next_ind, next_indT = get_ind_core(next_states)

    is_terminal = is_terminal_core(next_states, legal_move_mask, next_ind, next_indT)

    return next_states, is_terminal, next_ind, next_indT


def select_indices(probs, random_values):
    return (probs.cumsum(dim=1) < random_values.unsqueeze(-1)).sum(dim=1)


def row_can_move_left(row):
    if any(i == j and i != 0 for i, j in zip(row[:-1], row[1:])):
        # A merge can be done
        return True

    first_zero = next((i for i, x in enumerate(row) if x == 0), None)
    last_non_zero = next((len(row) - i for i, x in enumerate(reversed(row)) if x != 0), None)
    if first_zero is not None and last_non_zero is not None and first_zero < last_non_zero:
        # The first zero is to the left of the last non-zero, so the row can shift over
        return True

    return False


def row_can_move_right(row):
    return row_can_move_left(np.flip(row, 0))


def shift_row_left(row) -> int:
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


@cache
def get_move_left_map_and_rewards():
    move_left_map = torch.zeros((16**4, 4), dtype=torch.uint8)
    reward_map = torch.zeros((16**4, 1), dtype=torch.float32)
    lookup_tensor = torch.tensor([16**3, 16**2, 16, 1], dtype=torch.int32)
    for i in range(16):
        for j in range(16):
            for k in range(16):
                for l in range(16):
                    row = torch.tensor([i, j, k, l], dtype=torch.int32)
                    ind = int(lookup_tensor.dot(row).item())
                    reward = shift_row_left(row)
                    move_left_map[ind, :] = row
                    reward_map[ind] = reward

    return move_left_map, reward_map


@cache
def get_move_right_map():
    move_right_map = torch.zeros((16**4, 4), dtype=torch.uint8)
    lookup_tensor = torch.tensor([16**3, 16**2, 16, 1], dtype=torch.int32)
    for i in range(16):
        for j in range(16):
            for k in range(16):
                for l in range(16):
                    row = torch.tensor([i, j, k, l], dtype=torch.int32)
                    ind = int(lookup_tensor.dot(row).item())
                    row = torch.flip(row, [0])
                    shift_row_left(row)
                    row = torch.flip(row, [0])
                    move_right_map[ind, :] = row

    return move_right_map


@cache
def get_legal_move_masks():
    """
    Fills in the LEGAL_MOVE_MASK array for a row. The format is [LEFT, RIGHT] (use transposes to get up down)
    """
    legal_move_mask = torch.zeros((16**4, 2), dtype=torch.bool)
    lookup_tensor = torch.tensor([16**3, 16**2, 16, 1], dtype=torch.int32)
    for i in range(16):
        for j in range(16):
            for k in range(16):
                for l in range(16):
                    row = torch.tensor([i, j, k, l], dtype=torch.int32)
                    ind = int(lookup_tensor.dot(row).item())
                    legal_move_mask[ind, 0] = row_can_move_left(row)
                    legal_move_mask[ind, 1] = row_can_move_left(torch.flip(row, [0]))

    return legal_move_mask


class G2048Multi(GameMulti):

    device = 'cpu'

    MOVE_LEFT_MAP, REWARD_MAP = get_move_left_map_and_rewards()
    MOVE_LEFT_MAP = MOVE_LEFT_MAP.to(device)
    REWARD_MAP = REWARD_MAP.to(device)
    MOVE_RIGHT_MAP = get_move_right_map().to(device)
    LEGAL_MOVE_MASK = get_legal_move_masks().to(device)

    get_next_states_jit = torch.jit.trace(
        get_next_states_full,
        example_inputs=(torch.randint(0, 2, (2, 4, 4), dtype=torch.uint8, device=device),
                        torch.zeros((2,), dtype=torch.long, device=device),
                        torch.randint(0, 2, (16 ** 4, 4), dtype=torch.uint8, device=device),
                        torch.randint(0, 2, (16 ** 4, 4), dtype=torch.uint8, device=device),
                        torch.randint(0, 2, (16 ** 4, 1), dtype=torch.float32, device=device),
                        torch.randint(0, 2, (16 ** 4, 2), dtype=torch.bool, device=device),
                        torch.randint(0, 2, (2, 4, 1), dtype=torch.long, device=device),
                        torch.randint(0, 2, (2, 1, 4), dtype=torch.long, device=device))
    )

    get_legal_action_masks_jit = torch.jit.trace(
        get_legal_action_masks_core,
        example_inputs=(
            torch.randint(0, 2, (2, 4, 4), dtype=torch.uint8, device=device),
            torch.randint(0, 2, (16 ** 4, 2), dtype=torch.bool, device=device),
            torch.randint(0, 2, (2, 4, 1), dtype=torch.long, device=device),
            torch.randint(0, 2, (2, 1, 4), dtype=torch.long, device=device))
    )

    is_terminal_jit = torch.jit.trace(
        is_terminal_core,
        example_inputs=(
            torch.randint(0, 2, (2, 4, 4), dtype=torch.uint8, device=device),
            torch.randint(0, 2, (16 ** 4, 2), dtype=torch.bool, device=device),
            torch.randint(0, 2, (2, 4, 1), dtype=torch.long, device=device),
            torch.randint(0, 2, (2, 1, 4), dtype=torch.long, device=device))
    )

    get_next_states_from_env_jit = torch.jit.trace(
        get_next_states_from_env_core,
        example_inputs=(torch.randint(0, 2, (2, 4, 4), dtype=torch.uint8, device=device),
                        torch.rand(2, dtype=torch.float32, device=device),
                        torch.randint(0, 2, (16 ** 4, 2), dtype=torch.bool, device=device))
    )

    get_next_states_from_env_action_idx_jit = torch.jit.trace(
        get_next_states_from_env_action_idx,
        example_inputs=(torch.randint(0, 2, (2, 4, 4), dtype=torch.uint8, device=device),
                        torch.randint(0, 2, (2,), dtype=torch.long, device=device),
                        torch.randint(0, 2, (16 ** 4, 2), dtype=torch.bool, device=device))
    )

    get_env_legal_action_masks_jit = torch.jit.trace(
        get_env_legal_action_masks_core,
        example_inputs=(torch.randint(0, 2, (2, 4, 4), dtype=torch.uint8, device=device))
    )

    def __init__(self, n_parallel_games, player, listeners=None):
        super().__init__(n_parallel_games, player, listeners)

    @classmethod
    def get_n_players(cls):
        return 1

    @classmethod
    def get_n_actions(cls):
        return 4

    @classmethod
    def get_n_stochastic_actions(cls):
        return 32

    @classmethod
    def get_state_shape(cls) -> Tuple[int, ...]:
        return 4, 4

    @classmethod
    def get_state_dtype(cls) -> torch.dtype:
        return torch.uint8

    class Moves(enum.Enum):
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3

    @classmethod
    def is_terminal(cls, states: torch.FloatTensor):
        # If any zeros, not terminal, else if no legal actions, terminal
        # The shape of states is N x state shape so N x 4 x 4
        # We need to return bools of size (N,)
        return cls.is_terminal_jit(states, cls.LEGAL_MOVE_MASK, *get_ind(states))

    def get_cur_player_index(self, states) -> torch.Tensor:
        return torch.zeros((states.shape[0],), dtype=torch.long, device=states.device)

    @classmethod
    def get_next_states(cls, states: torch.Tensor, actions: torch.Tensor):
        """
        :param states: A tensor of size (N, 4, 4) representing the states
        :param actions: A tensor of size (N,) representing the actions
        """
        new_states, rewards, env_is_next, is_terminal, new_ind, new_indT = (
            cls.get_next_states_jit(states, actions, cls.MOVE_LEFT_MAP, cls.MOVE_RIGHT_MAP,
                                    cls.REWARD_MAP, cls.LEGAL_MOVE_MASK, *get_ind(states))
        )

        new_states.ind = new_ind
        new_states.new_indT = new_indT

        return new_states, rewards, env_is_next, is_terminal

    @classmethod
    def get_env_legal_action_masks(cls, states):
        return cls.get_env_legal_action_masks_jit(states)

    @classmethod
    def get_next_states_from_env(cls, states: torch.Tensor, action_idx=None):
        """
        For each state in the states tensor, we need to add a random 1 or 2 into a slot that is zero

        :param states: A tensor of size (N, 4, 4) representing the states
        :param legal_action_mask: a pre-computed mask to use if passed
        """
        if action_idx is not None:
            next_states, is_terminal, next_ind, next_indT = cls.get_next_states_from_env_action_idx_jit(states, action_idx, cls.LEGAL_MOVE_MASK)
        else:
            action_idx = cls.get_env_action_idx(states, cls.get_env_legal_action_masks(states))
            return cls.get_next_states_from_env(states, action_idx)

        next_states.ind = next_ind
        next_states.indT = next_indT

        return next_states, is_terminal

    @classmethod
    def get_env_action_idx(cls, states, legal_action_mask):
        pi = torch.tensor([9 for _ in range(16)] + [1 for _ in range(16)], dtype=torch.float32)
        pi = pi.repeat(states.shape[0], 1)
        pi *= legal_action_mask
        pi /= pi.sum(dim=1, keepdim=True)
        return torch.multinomial(pi, 1).flatten()

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

    @classmethod
    def get_initial_states(cls, n_games):
        states = torch.zeros((n_games, 4, 4), dtype=torch.uint8, device=cls.device)

        # Add in two random values for each state
        states, _ = cls.get_next_states_from_env(states)
        states, _ = cls.get_next_states_from_env(states)

        return states

    @classmethod
    def get_legal_action_masks(cls, states: torch.FloatTensor):
        """
        :param states: A tensor of size (N, 4, 4) representing the states
        """
        return cls.get_legal_action_masks_jit(states, cls.LEGAL_MOVE_MASK, *get_ind(states))

    @classmethod
    def delete_jit_methods(cls):
        del cls.get_next_states_jit
        del cls.get_legal_action_masks_jit
        del cls.is_terminal_jit
        del cls.get_next_states_from_env_jit

    def __str__(self):
        import termcolor
        color_map = {
            0: 'dark_grey',
            1: 'light_grey',
            2: 'white',
            3: 'light_yellow',
            4: 'yellow',
            5: 'light_red',
            6: 'red',
            7: 'light_blue',
            8: 'blue',
            9:  'light_magenta',
            10: 'magenta',
            11: 'light_green',
            12: 'green',
            13: 'green',
            14: 'green',
            15: 'green',
            16: 'green'
        }

        def line_to_str(line, padding):
            out = []
            for i in line:
                num = str(2**i.item()) if i.item() > 0 else ''
                out.append(termcolor.colored(num.ljust(padding), color=color_map.get(i.item(), 'white')))

            return out

        def grid_to_str(grid, padding):
            x = ' ' * padding
            return ('-' * line_length + '\n').join(
                [(f' {x} | {x} | {x} | {x} \n' + ' {} | {} | {} | {} \n' + f' {x} | {x} | {x} | {x} \n').format(*line_to_str(grid[i, :], padding)) for i in range(grid.shape[0])])

        grid = self.states.to(int)
        padding = len(str(2**grid.max().item()))
        line_length = (padding + 2)*4 + 3
        return (
            ('-' * line_length + '\n' + '-' * line_length + '\n').join([grid_to_str(g, padding) for g in grid])
        )


class G2048MultiCuda(G2048Multi):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    MOVE_LEFT_MAP, REWARD_MAP = get_move_left_map_and_rewards()
    MOVE_LEFT_MAP = MOVE_LEFT_MAP.to(device)
    REWARD_MAP = REWARD_MAP.to(device)
    MOVE_RIGHT_MAP = get_move_right_map().to(device)
    LEGAL_MOVE_MASK = get_legal_move_masks().to(device)

    get_next_states_jit = torch.jit.trace(
        get_next_states_full,
        example_inputs=(torch.randint(0, 2, (2, 4, 4), dtype=torch.uint8, device=device),
                        torch.zeros((2,), dtype=torch.long, device=device),
                        torch.randint(0, 2, (16 ** 4, 4), dtype=torch.uint8, device=device),
                        torch.randint(0, 2, (16 ** 4, 4), dtype=torch.uint8, device=device),
                        torch.randint(0, 2, (16 ** 4, 1), dtype=torch.float32, device=device),
                        torch.randint(0, 2, (16 ** 4, 2), dtype=torch.bool, device=device),
                        torch.randint(0, 2, (2, 4, 1), dtype=torch.long, device=device),
                        torch.randint(0, 2, (2, 1, 4), dtype=torch.long, device=device))
    )

    get_legal_action_masks_jit = torch.jit.trace(
        get_legal_action_masks_core,
        example_inputs=(
            torch.randint(0, 2, (2, 4, 4), dtype=torch.uint8, device=device),
            torch.randint(0, 2, (16 ** 4, 2), dtype=torch.bool, device=device),
            torch.randint(0, 2, (2, 4, 1), dtype=torch.long, device=device),
            torch.randint(0, 2, (2, 1, 4), dtype=torch.long, device=device))
    )

    is_terminal_jit = torch.jit.trace(
        is_terminal_core,
        example_inputs=(
            torch.randint(0, 2, (2, 4, 4), dtype=torch.uint8, device=device),
            torch.randint(0, 2, (16 ** 4, 2), dtype=torch.bool, device=device),
            torch.randint(0, 2, (2, 4, 1), dtype=torch.long, device=device),
            torch.randint(0, 2, (2, 1, 4), dtype=torch.long, device=device))
    )

    get_next_states_from_env_jit = torch.jit.trace(
        get_next_states_from_env_core,
        example_inputs=(torch.randint(0, 2, (2, 4, 4), dtype=torch.uint8, device=device),
                        torch.rand(2, dtype=torch.float32, device=device),
                        torch.randint(0, 2, (16 ** 4, 2), dtype=torch.bool, device=device))
    )

    get_next_states_from_env_action_idx_jit = torch.jit.trace(
        get_next_states_from_env_action_idx,
        example_inputs=(torch.randint(0, 2, (2, 4, 4), dtype=torch.uint8, device=device),
                        torch.randint(0, 2, (2,), dtype=torch.long, device=device),
                        torch.randint(0, 2, (16 ** 4, 2), dtype=torch.bool, device=device))
    )

    get_env_legal_action_masks_jit = torch.jit.trace(
        get_env_legal_action_masks_core,
        example_inputs=(torch.randint(0, 2, (2, 4, 4), dtype=torch.uint8, device=device))
    )


def get_G2048Multi_game_class(device):
    if device == 'cuda':
        return G2048MultiCuda
    else:
        return G2048Multi
