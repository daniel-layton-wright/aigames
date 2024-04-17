from typing import Tuple
import torch


def is_terminal_and_rewards(states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    kernels = torch.tensor([
        [[0, 0, 0, 0],
         [1, 1, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 1, 0, 0],
         [0, 1, 0, 0],
         [0, 1, 0, 0],
         [0, 1, 0, 0]],
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        [[0, 0, 0, 1],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [1, 0, 0, 0]]
    ], dtype=torch.float32, device='cpu')

    conv_result = torch.nn.functional.conv2d(states[:, 1:, :].unsqueeze(1).to(torch.float32), kernels.unsqueeze(1), padding=2)
    player0_won = (conv_result == 4).any(dim=-1).any(dim=-1).any(dim=-1)
    player1_won = (conv_result == -4).any(dim=-1).any(dim=-1).any(dim=-1)
    no_more_moves = (states[:, 0].sum(dim=-1) == 42)

    is_terminal = player0_won | player1_won | no_more_moves

    rewards = torch.zeros((states.shape[0], 2), dtype=torch.float32, device='cpu')
    rewards[player0_won, 0] = 1
    rewards[player1_won, 0] = -1
    rewards[:, 1] = -rewards[:, 0]

    return is_terminal, rewards


is_terminal_and_rewards_jit = torch.jit.trace(is_terminal_and_rewards, (torch.randint(2, (2, 6, 7))))


def is_winning_state(bit_grid):
    # Check horizontal 4 in a row
    horizontal_neighbor = bit_grid & (bit_grid >> 7)
    if (horizontal_neighbor & (horizontal_neighbor >> 14)) > 0:
        # If a square has a horizontal neighbor and the square two over also has a horizontal neighbor,
        #   then there is 4 in a row
        return True

    # Check vertical 4 in a row
    vertical_neighbor = bit_grid & (bit_grid >> 1)
    if (vertical_neighbor & (vertical_neighbor >> 2)) > 0:
        return True

    # Check diagonal 4 in a row
    diagonal_neighbor = bit_grid & (bit_grid >> 8)
    if (diagonal_neighbor & (diagonal_neighbor >> 16)) > 0:
        return True

    # Check other diagonal 4 in a row
    other_diagonal_neighbor = bit_grid & (bit_grid >> 6)
    if (other_diagonal_neighbor & (other_diagonal_neighbor >> 12)) > 0:
        return True

    # If no 4 in a row found, return False
    return False


def is_terminal_and_rewards_bit(states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rewards = torch.zeros((states.shape[0], 2), dtype=torch.float32, device='cpu')
    is_terminal = torch.zeros((states.shape[0],), dtype=torch.bool, device='cpu')

    for i, state in enumerate(states):
        game_grid = state[0]
        player0_grid = state[1]
        player1_grid = game_grid ^ player0_grid
        player0_win = is_winning_state(player0_grid)
        player1_win = is_winning_state(player1_grid)
        is_terminal[i] = player0_win | player1_win | (state[2:].sum() == 42)
        rewards[i] = 1*player0_win - 1*player1_win

    return is_terminal, rewards


# is_terminal_and_rewards_bit_jit = torch.jit.trace(is_terminal_and_rewards_bit, (torch.randint(2, (1000, 9), dtype=torch.long),))


def main():
    import perftester as pt

    random_states = torch.randint(-1, 2, (1000, 6, 7), dtype=torch.int8)
    random_states[:, 0] = random_states.sum(dim=1)

    v1_results = pt.time_benchmark(is_terminal_and_rewards,
                                   states=random_states,
                                   Number=500, Repeat=1)

    v1_jit_results = pt.time_benchmark(is_terminal_and_rewards_jit,
                                       states=random_states,
                                       Number=500, Repeat=1)

    random_states = torch.randint(2, (1000, 9), dtype=torch.long)

    vbit_results = pt.time_benchmark(is_terminal_and_rewards_bit,
                                   states=random_states,
                                   Number=500, Repeat=1)

    # vbit_jit_results = pt.time_benchmark(is_terminal_and_rewards_bit_jit,
    #                                    states=random_states,
    #                                    Number=500, Repeat=1)

    print('v1:')
    print(pt.pp(v1_results))
    print('v1_jit:')
    print(pt.pp(v1_jit_results))

    print('vbit:')
    print(pt.pp(vbit_results))
    print('vbit_jit:')
    print(pt.pp(vbit_jit_results))


if __name__ == '__main__':
    main()