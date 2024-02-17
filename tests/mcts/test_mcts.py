import unittest
import torch
from aigames.agent.alpha_agent_multi import AlphaAgentMulti, AlphaAgentHyperparametersMulti
from aigames.agent.alpha_agent_multi import DummyAlphaEvaluatorMulti
from aigames.mcts.mcts import MCTS
from aigames.game.G2048_multi import G2048Multi


class TestMCTS(unittest.TestCase):
    def test_mcts(self):
        # Seed
        torch.manual_seed(1236796)

        hyperparams = AlphaAgentHyperparametersMulti()
        hyperparams.n_mcts_iters = 100
        hyperparams.c_puct = 1
        hyperparams.dirichlet_alpha = 1
        hyperparams.dirichlet_epsilon = 0.25
        hyperparams.discount = 0.99
        hyperparams.expand_simultaneous_fraction = 1.0
        hyperparams.dirichlet_epsilon = 0.

        dummy_evaluator = DummyAlphaEvaluatorMulti(4, 1, 'cpu')
        agent = AlphaAgentMulti(G2048Multi, dummy_evaluator, hyperparams)
        game = G2048Multi(100, agent)

        mcts = MCTS(game, dummy_evaluator, hyperparams, G2048Multi.get_initial_states(100))

        mcts.search_for_n_iters(100)

        print(mcts.n)

        import pdb
        pdb.set_trace()
