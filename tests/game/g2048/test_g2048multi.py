import unittest
import torch
from aigames.agent.random_agent_multi import RandomAgentMulti
from aigames.game.G2048_multi import G2048Multi


class TestG2048Multi(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_parallel_games = 2
        self.agent = RandomAgentMulti(G2048Multi)
        self.game = G2048Multi(self.n_parallel_games, self.agent)

    def test_get_next_states(self):
        states = torch.FloatTensor([[[1, 0, 1, 0],
                                     [0, 0, 0, 0],
                                     [2, 0, 3, 0],
                                     [4, 4, 4, 3]],
                                    [[0, 0, 1, 1],
                                     [1, 1, 1, 1],
                                     [2, 0, 0, 2],
                                     [0, 0, 3, 2]]])

        actions = torch.LongTensor([0, 2])

        next_states, rewards, is_env, is_terminal = self.game.get_next_states(states, actions)

        expected_next_states = torch.FloatTensor([[[2, 0, 0, 0],
                                                   [0, 0, 0, 0],
                                                   [2, 3, 0, 0],
                                                   [5, 4, 3, 0]],
                                                 [[1, 1, 2, 2],
                                                  [2, 0, 3, 3],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]]])

        expected_reward = torch.FloatTensor([[2**2 + 2**5], [2**2 + 2**2 + 2**3]])

        expected_is_env = torch.BoolTensor([True, True])

        expected_is_terminal = torch.BoolTensor([False, False])

        torch.testing.assert_close(next_states, expected_next_states)
        torch.testing.assert_close(rewards, expected_reward)
        torch.testing.assert_close(is_env, expected_is_env)
        torch.testing.assert_close(is_terminal, expected_is_terminal)
