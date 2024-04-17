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
        states = torch.tensor([[[1, 0, 1, 0],
                                     [0, 0, 0, 0],
                                     [2, 0, 3, 0],
                                     [4, 4, 4, 3]],
                                    [[0, 0, 1, 1],
                                     [1, 1, 1, 1],
                                     [2, 0, 0, 2],
                                     [0, 0, 3, 2]]], dtype=torch.uint8)

        actions = torch.LongTensor([0, 2])

        next_states, rewards, is_env, is_terminal = self.game.get_next_states(states, actions)

        expected_next_states = torch.tensor([[[2, 0, 0, 0],
                                                   [0, 0, 0, 0],
                                                   [2, 3, 0, 0],
                                                   [5, 4, 3, 0]],
                                                 [[1, 1, 2, 2],
                                                  [2, 0, 3, 3],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]]], dtype=torch.uint8)

        expected_reward = torch.FloatTensor([[2**2 + 2**5], [2**2 + 2**2 + 2**3]])

        expected_is_env = torch.BoolTensor([True, True])

        expected_is_terminal = torch.BoolTensor([False, False])

        torch.testing.assert_close(next_states, expected_next_states)
        torch.testing.assert_close(rewards, expected_reward)
        torch.testing.assert_close(is_env, expected_is_env)
        torch.testing.assert_close(is_terminal, expected_is_terminal)

    def test_g2048_get_env_legal_action_masks_core(self):
        from aigames.game.G2048_multi import get_env_legal_action_masks_core

        states = torch.tensor([[[1, 0, 1, 0],
                                     [0, 0, 0, 0],
                                     [2, 0, 3, 0],
                                     [4, 4, 4, 3]],
                                    [[0, 0, 1, 1],
                                     [1, 1, 1, 1],
                                     [2, 0, 0, 2],
                                     [0, 0, 3, 2]]], dtype=torch.uint8)

        env_legal_actions_masks = get_env_legal_action_masks_core(states)

        expected_legal_action_masks = torch.BoolTensor([
            [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0]
        ])

        torch.testing.assert_close(env_legal_actions_masks, expected_legal_action_masks)
