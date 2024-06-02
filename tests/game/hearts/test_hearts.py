"""
Tests for correct functionality of the hearts game
"""

import unittest

from aigames import GameListenerMulti
from aigames.agent.alpha_agent_multi import DummyAlphaEvaluatorMulti
from aigames.agent.hearts.simple_hearts_agent import SimpleHeartsAgent
from aigames.agent.random_agent_multi import RandomAgentMulti
from aigames.game.hearts import Hearts
import torch

from aigames.utils.listeners import RewardListenerMulti, ActionCounterProgressBar


class TestHearts(unittest.TestCase):
    def test_initial_states_have_correct_form(self):
        states = Hearts.get_initial_states(4)
        self.assertEqual(states.shape, (4, 5, 55))

        # Each player should have 13 cards
        self.assertTrue(((states[:, 0, 3:] == 1).sum(axis=-1) == 13).all())
        self.assertTrue(((states[:, 0, 3:] == 2).sum(axis=-1) == 13).all())
        self.assertTrue(((states[:, 0, 3:] == 3).sum(axis=-1) == 13).all())
        self.assertTrue(((states[:, 0, 3:] == 4).sum(axis=-1) == 13).all())

        # legal actions should all only allow two of clubs
        legal_action_masks = Hearts.get_legal_action_masks(states)

        self.assertTrue((legal_action_masks[:, 0] == 1).all())

    def test_get_next_states(self):
        states = Hearts.get_initial_states(4)
        actions = torch.tensor([0, 0, 0, 0])
        next_states, rewards, is_env, is_terminal = Hearts.get_next_states(states, actions)

        # Check that the rewards are all zero
        self.assertTrue((rewards == 0).all())

        # Check that is_env is all false
        self.assertTrue((is_env == False).all())

        # Check that is_terminal is all false
        self.assertTrue((is_terminal == False).all())

    def test_end_of_trick(self):
        # Get initial states and then advance to the end of the first trick by taking the first legal action for
        # each of the four players

        states = Hearts.get_initial_states(4)

        for _ in range(4):
            # Get legal actions in next state and pick the first one for each player
            print(Hearts.state_to_str(states[0]))
            legal_action_masks = Hearts.get_legal_action_masks(states)
            actions = legal_action_masks.to(torch.int8).argmax(dim=-1)
            states, rewards, is_env, is_terminal = Hearts.get_next_states(states, actions)

        print(Hearts.state_to_str(states[0]))

        # check no cards played to current trick
        self.assertTrue((states[:, 4, 3:] == 0).all())
        self.assertTrue(states[0, 0, 1] == 0)

    def test_shooting_the_moon_score(self):
        state = Hearts.get_initial_states(1)

        # Now we overwrite the initial state so that someone is going to shoot the moon
        # Player 1 gets all the clubs so that they will always have the lead
        state[0, 0, 3:16] = 1
        state[0, 0, 0] = 1

        # The rest can be random
        state[0, 0, 16:55] = torch.arange(2, 5).repeat(13)[torch.randperm(39)]

        class StateOverwriter(GameListenerMulti):
            def before_game_start(self, game):
                game.states = state

        reward_listener = RewardListenerMulti(1)
        game = Hearts(1, SimpleHeartsAgent(), [reward_listener, StateOverwriter()])
        game.play()

        # Check that the rewards are [0, -26, -26, -26]
        torch.testing.assert_close(reward_listener.rewards, torch.tensor([0, -26, -26, -26], dtype=torch.float32).unsqueeze(0))

    def test_score_sums(self):
        # Play a bunch of games and check the scores. They should sum to -26 or (if someone shoots the moon) -78
        n_games = 100
        reward_listener = RewardListenerMulti(1)
        game = Hearts(n_games, RandomAgentMulti(Hearts), [reward_listener])
        game.play()

        sum_per_game = reward_listener.rewards.sum(dim=1)

        # check that the sum is either -26 or -78
        self.assertTrue(((sum_per_game == -26) | (sum_per_game == -78)).all())

    def test_alpha_agent_works(self):
        """
        Make sure the game is playable with the AlphaAgentMulti and a dummy evaluator (essentially checking mcts works)
        """
        from aigames.agent.alpha_agent_multi import AlphaAgentMulti, AlphaAgentHyperparametersMulti

        eval = DummyAlphaEvaluatorMulti(52, 4)
        hypers = AlphaAgentHyperparametersMulti()
        alpha_agent = AlphaAgentMulti(Hearts, eval, hypers)

        game = Hearts(1, alpha_agent,
                      [ActionCounterProgressBar(52, 'test_alpha_agent_works')])
        game.play()
