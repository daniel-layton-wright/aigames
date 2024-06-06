import unittest
import torch
from aigames.agent.alpha_agent_multi import AlphaAgentMulti, AlphaAgentHyperparametersMulti, ConstantMCTSIters
from aigames.agent.alpha_agent_multi import DummyAlphaEvaluatorMulti
from aigames.mcts.mcts import MCTS
from aigames.game.G2048_multi import G2048Multi
from ..helpers import Timer


class TestMCTS(unittest.TestCase):
    def test_mcts(self):
        # Seed
        torch.manual_seed(1236796)

        hyperparams = AlphaAgentHyperparametersMulti()
        hyperparams.n_mcts_iters = ConstantMCTSIters(100)
        hyperparams.c_puct = 1
        hyperparams.dirichlet_alpha = 1
        hyperparams.dirichlet_epsilon = 0.25
        hyperparams.discount = 0.99
        hyperparams.expand_simultaneous_fraction = 1.0
        hyperparams.dirichlet_epsilon = 0.

        dummy_evaluator = DummyAlphaEvaluatorMulti(4, 1, 'cpu')
        agent = AlphaAgentMulti(G2048Multi, dummy_evaluator, hyperparams)
        game = G2048Multi(100, agent)

        mcts = MCTS(game, dummy_evaluator, hyperparams, hyperparams.n_mcts_iters.get_n_mcts_iters()[0],
                    G2048Multi.get_initial_states(100))

        mcts.search_for_n_iters(100)

        print(mcts.n)

    def test_get_next_mcts(self):
        # Seed
        torch.manual_seed(1236796)

        hyperparams = AlphaAgentHyperparametersMulti()
        hyperparams.n_mcts_iters = ConstantMCTSIters(100)
        hyperparams.c_puct = 1
        hyperparams.dirichlet_alpha = 1
        hyperparams.dirichlet_epsilon = 0.25
        hyperparams.discount = 0.99
        hyperparams.expand_simultaneous_fraction = 1.0
        hyperparams.dirichlet_epsilon = 0.

        dummy_evaluator = DummyAlphaEvaluatorMulti(4, 1, 'cpu')
        agent = AlphaAgentMulti(G2048Multi, dummy_evaluator, hyperparams)
        game = G2048Multi(1000, agent)

        states = G2048Multi.get_initial_states(1000)
        mcts = MCTS(game, dummy_evaluator, hyperparams, hyperparams.n_mcts_iters.get_n_mcts_iters()[0], states)

        mcts.search_for_n_iters(100)

        # Advance the game now with a move and an env move
        # For the actions take the argmax of mcts.n
        actions = mcts.n[mcts.root_idx, mcts.root_nodes].argmax(dim=1)
        states, rewards, is_env, is_term = game.get_next_states(states, actions)
        states, is_term = game.get_next_states_from_env(states)

        # Gather data to check results after doing mcts.get_next_mcts
        state_match = mcts.states[mcts.root_idx] == states.unsqueeze(1)
        for _ in range(len(mcts.state_shape)):
            state_match = state_match.all(dim=-1)

        state_match *= (~mcts.is_env[mcts.root_idx])

        new_root_nodes = state_match.int().argmax(dim=1)
        idx = mcts.root_idx[new_root_nodes > 0]
        existing_roots = new_root_nodes[new_root_nodes > 0]

        pi = mcts.pi[idx, existing_roots].clone()
        n = mcts.n[idx, existing_roots].clone()
        w = mcts.w[idx, existing_roots].clone()
        next_idx = mcts.next_idx[idx, existing_roots].clone()

        mcts = mcts.get_next_mcts(states)

        # Check that the data for the roots of the new mcts match the old tree
        torch.testing.assert_close(mcts.pi[idx, 1], pi)
        torch.testing.assert_close(mcts.n[idx, 1], n)
        torch.testing.assert_close(mcts.w[idx, 1], w)
        torch.testing.assert_close(mcts.next_idx[idx, 1], next_idx)

        # The parent of the children should match
        children = mcts.next_idx[idx, 1]
        parents = mcts.parent_nodes[idx.unsqueeze(-1).repeat(1, children.shape[1]), children]
        # If the children was not 0 then the corresponding parent should be 1
        parents += 1*(children == 0)
        torch.testing.assert_close(parents, torch.ones_like(parents))

        # We don't want to reuse any env nodes even if they match
        torch.testing.assert_close(mcts.is_env[idx, 1], torch.zeros_like(mcts.is_env[idx, 1]))

        # Testing for leaf roots
        leaf_root_idx = mcts.root_idx[new_root_nodes == 0]
        leaf_root_nodes = new_root_nodes[new_root_nodes == 0]

        # The n[:, 0, 0] should be equal to n[:, 1].sum(dim=1)
        torch.testing.assert_close(mcts.n[:, 0, 0].to(torch.int64), mcts.n[:, 1].sum(dim=1))

        # is leaf should be true
        self.assertTrue(mcts.is_leaf[leaf_root_idx, leaf_root_nodes].all())

        # n should be 0
        torch.testing.assert_close(mcts.n[leaf_root_idx, 1],
                                   torch.zeros_like(mcts.n[leaf_root_idx, leaf_root_nodes]))

        # w should be 0
        torch.testing.assert_close(mcts.w[leaf_root_idx, 1],
                                   torch.zeros_like(mcts.w[leaf_root_idx, leaf_root_nodes]))
