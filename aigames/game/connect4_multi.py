from typing import Union, List, Tuple
import torch
from aigames import AgentMulti, GameListenerMulti
from aigames.game.game_multi import GameMulti


class Connect4Multi(GameMulti):
    device = 'cpu'

    def __init__(self, n_parallel_games, players: Union[AgentMulti, List[AgentMulti]],
                 listeners: List[GameListenerMulti] = None):
        super().__init__(n_parallel_games, players, listeners)

    @classmethod
    def get_n_players(cls) -> int:
        return 2

    @classmethod
    def get_n_actions(cls) -> int:
        return 7

    @classmethod
    def get_state_shape(cls) -> Tuple[int, ...]:
        return 7, 7

    @classmethod
    def get_n_stochastic_actions(cls) -> int:
        return 0

    @classmethod
    def get_state_dtype(cls) -> torch.dtype:
        return torch.int8

    @classmethod
    def get_initial_states(cls, n) -> torch.Tensor:
        return torch.zeros((n, 6 + 1, 7), dtype=torch.int8, device=cls.device)

    def get_cur_player_index(self, states) -> torch.Tensor:
        return states[:, 0].sum(dim=1) % 2

    def get_next_states(self, states: torch.Tensor, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        next_states = states.clone()
        cur_player = self.get_cur_player_index(states)
        player_mark = -2*cur_player + 1  # 0 -> 1; 1 -> -1
        r = torch.arange(states.shape[0])
        next_states[r, 6 - states[r, 0, actions].long(), actions] = player_mark.to(torch.int8)
        next_states[r, 0, actions] += 1

        is_terminal, rewards = self.is_terminal_and_rewards(next_states)

        return next_states, rewards, torch.zeros(states.shape[0], dtype=torch.bool, device=self.device), is_terminal

    @classmethod
    def is_terminal(cls, states):
        return cls.is_terminal_and_rewards(states)[0]

    @classmethod
    def is_terminal_and_rewards(cls, states) -> Tuple[torch.Tensor, torch.Tensor]:
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
        ], dtype=torch.float32, device=cls.device)

        conv_result = torch.nn.functional.conv2d(states[:, 1:, :].unsqueeze(1).to(torch.float32), kernels.unsqueeze(1), padding=2)
        player0_won = (conv_result == 4).any(dim=-1).any(dim=-1).any(dim=-1)
        player1_won = (conv_result == -4).any(dim=-1).any(dim=-1).any(dim=-1)
        no_more_moves = (states[:, 0].sum(dim=-1) == 42)

        is_terminal = player0_won | player1_won | no_more_moves

        rewards = torch.zeros((states.shape[0], 2), dtype=torch.float32, device=cls.device)
        rewards[player0_won, 0] = 1
        rewards[player1_won, 0] = -1
        rewards[:, 1] = -rewards[:, 0]

        return is_terminal, rewards

    def get_legal_action_masks(self, states):
        return states[:, 0] < 6

    def __str__(self):
        import termcolor

        color_map = {
            1: 'red',
            -1: 'blue'
        }

        def val_to_str(val):
            if val == 1 or val == -1:
                return '⬤'
            else:
                return ' '

        def line_to_str(line, padding) -> str:
            out = []
            for i in line:
                num = val_to_str(i)
                out.append(termcolor.colored(num.ljust(padding), color=color_map.get(i.item(), 'white')))

            out = '|'.join(out)

            return out

        def grid_to_str(grid):
            out = ''
            for row in grid[1:]:
                out += line_to_str(row, 2) + '\n'

            # overbar character
            out += '⎺' * (len(grid[0]) * (2 + 1) - 1) + '\n'

            return out

        padding = 1
        line_length = (padding + 2)*7 + 3
        return (
            ('-' * line_length + '\n' + '-' * line_length + '\n').join([grid_to_str(g) for g in self.states])
        )
