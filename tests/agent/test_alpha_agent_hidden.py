import unittest
import torch
from aigames.agent.alpha_agent_hidden import AlphaAgentHidden, AlphaAgentHiddenHyperparameters
from aigames.agent.hearts.hearts_sampler import HeartsSampler
from aigames.game.game_multi import GameMulti
from aigames.game.hearts_hidden import HeartsHidden
from aigames.agent.alpha_agent_multi import DummyAlphaEvaluatorMulti
from aigames.utils.listeners import RewardListenerMulti


class TestAlphaAgentHidden(unittest.TestCase):
    """
    Test cases for the AlphaAgentHidden using the HeartsSampler.
    """

    def setUp(self):
        """
        Set up the test environment by initializing the game, sampler, evaluator,
        hyperparameters, and the AlphaAgentHidden instance.
        """
        # Initialize the Hearts game class
        self.game_class: GameMulti = HeartsHidden

        # Initialize the HeartsSampler
        self.sampler = HeartsSampler()

        # Initialize a dummy evaluator
        self.evaluator = DummyAlphaEvaluatorMulti(n_actions=self.game_class.get_n_actions(),
                                                  n_players=self.game_class.get_n_players(),
                                                  device='cpu')

        # Initialize hyperparameters
        self.hyperparams = AlphaAgentHiddenHyperparameters(n_samples=10)

        # Initialize listeners
        self.listeners = [RewardListenerMulti(1)]

        # Initialize the AlphaAgentHidden
        self.alpha_agent_hidden = AlphaAgentHidden(game_class=self.game_class,
                                                   evaluator=self.evaluator,
                                                   hyperparams=self.hyperparams,
                                                   listeners=self.listeners,
                                                   sampler=self.sampler)
        
        self.alpha_agent_hidden.game = self.game_class

    def test_get_actions(self):
        """
        Test the get_actions method of AlphaAgentHidden to ensure it returns
        valid actions given a batch of states.
        """
        # Generate initial states using the game class
        n_games = 4
        states = self.game_class.get_initial_states(n_games)
        player_indices = self.game_class.get_cur_player_index(states)
        hidden_states = self.game_class.get_observable_states(states, player_indices)

        # Get actions from the AlphaAgentHidden
        actions = self.alpha_agent_hidden.get_actions(hidden_states, torch.ones(n_games).to(torch.bool))

        # Assert that actions are within the valid range
        self.assertTrue(torch.all(actions >= 0) and torch.all(actions < self.game_class.get_n_actions()),
                        msg="Actions contain invalid indices.")

        # Assert that the number of actions matches the number of states
        self.assertEqual(actions.shape[0], hidden_states.shape[0],
                         msg="Number of actions does not match number of states.")


if __name__ == '__main__':
    unittest.main()
