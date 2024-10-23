import unittest
import torch
from aigames.agent.alpha_agent_multi import AlphaAgentMulti, AlphaAgentHyperparametersMulti, ConstantMCTSIters
from aigames.agent.alpha_agent_multi import DummyAlphaEvaluatorMulti
from aigames.agent.random_agent_multi import RandomAgentMulti
from aigames.game.hearts import Hearts
from aigames.mcts.mcts import MCTS, UCBActionSelector, UCBHyperparameters, MCTSHyperparameters
from aigames.game.G2048_multi import G2048Multi
from ..helpers import Timer


class TestMCTS(unittest.TestCase):
    def test_mcts(self):
        # Seed
        torch.manual_seed(1236796)

        hyperparams = MCTSHyperparameters(
            dirichlet_alpha=1,
            dirichlet_epsilon=0,
            discount=0.99,
            expand_simultaneous_fraction=1.0,
            action_selector=UCBActionSelector(UCBHyperparameters(c_puct=1.0))
        )
        n_mcts_iters = ConstantMCTSIters(100)

        dummy_evaluator = DummyAlphaEvaluatorMulti(4, 1, 'cpu')
        agent = AlphaAgentMulti(G2048Multi, dummy_evaluator, hyperparams, n_mcts_iters)
        game = G2048Multi(100, agent)

        mcts = MCTS(game, dummy_evaluator, hyperparams, n_mcts_iters.get_n_mcts_iters()[0],
                    G2048Multi.get_initial_states(100))

        mcts.search_for_n_iters(100)

        print(mcts.n)

    def test_get_next_mcts(self):
        # Seed
        torch.manual_seed(1236796)

        hyperparams = MCTSHyperparameters(
            dirichlet_alpha=1,
            dirichlet_epsilon=0,
            discount=0.99,
            expand_simultaneous_fraction=1.0,
            action_selector=UCBActionSelector(UCBHyperparameters(c_puct=1.0))
        )
        n_mcts_iters = ConstantMCTSIters(100)

        dummy_evaluator = DummyAlphaEvaluatorMulti(4, 1, 'cpu')
        agent = AlphaAgentMulti(G2048Multi, dummy_evaluator, hyperparams, n_mcts_iters)
        game = G2048Multi(1000, agent)

        states = G2048Multi.get_initial_states(1000)
        mcts = MCTS(game, dummy_evaluator, hyperparams, n_mcts_iters.get_n_mcts_iters()[0], states)

        mcts.search_for_n_iters(100)

        # Advance the game now with a move and an env move
        # For the actions take the argmax of mcts.n
        actions = mcts.n[mcts.root_idx, 1].argmax(dim=1)
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

    def test_graph_from_mcts(self):
        from aigames.mcts.mcts import UCBActionSelector, UCBHyperparameters
        from aigames.mcts.utils import graph_from_mcts

        hyperparams = MCTSHyperparameters(
            dirichlet_alpha=1,
            dirichlet_epsilon=0.25,
            discount=1,
            expand_simultaneous_fraction=1.0,
            action_selector=UCBActionSelector(UCBHyperparameters(c_puct=1.0))
        )
        n_mcts_iters = ConstantMCTSIters(100)

        # Do an MCTS with Hearts
        hearts = Hearts(1, RandomAgentMulti(Hearts))
        states = Hearts.get_initial_states(1)
        states = Hearts.get_next_states(states, torch.tensor([0], dtype=torch.long))[0]

        states = torch.tensor([[1, 0, 36, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
                                0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                4, 0, 2, 0, 4, 3, 0, 0, 0, 3, 2, 4, 3, 1, 4, 2, 0, 0, 2],
                               [0, 0, 0, 1, 3, 2, 31, 0, 4, 21, 24, 23, 22, 29, 32, 0, 5, 6,
                                25, 19, 17, 8, 7, 26, 0, 0, 20, 27, 18, 12, 9, 10, 15, 11, 13, 14,
                                0, 16, 0, 28, 0, 0, 34, 35, 33, 0, 0, 0, 0, 0, 0, 0, 36, 30, 0],
                               [0, 0, 0, 1, 3, 2, 1, 0, 4, 4, 3, 2, 1, 3, 2, 0, 4, 1,
                                1, 1, 3, 3, 2, 2, 0, 0, 2, 3, 4, 1, 2, 3, 2, 4, 4, 1,
                                0, 3, 0, 4, 0, 0, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0],
                               [0, 0, 0, 4, 4, 4, 2, 0, 4, 1, 1, 1, 1, 2, 2, 0, 2, 2,
                                3, 4, 4, 2, 2, 3, 0, 0, 4, 3, 4, 4, 4, 4, 3, 4, 3, 3,
                                0, 3, 0, 3, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int8)
        states = states.unsqueeze(0)

        mcts = MCTS(hearts, DummyAlphaEvaluatorMulti(52, 4), hyperparams,
                    100, states)

        mcts.search_for_n_iters(100)
        graph = graph_from_mcts(mcts)
        graph.draw('mcts_graph.png', prog='dot')


class TestGumbelActionSelector(unittest.TestCase):
    """Tests for GumbelActionSelector."""

    def test_get_actions_roots_sequentially(self):
        """
        Test the `get_actions_roots` method of `GumbelActionSelector` using a dummy MCTS object.
        
        This test initializes a stubbed MCTS instance with necessary attributes and verifies that
        `get_actions_roots` produces valid action indices across multiple iterations while manually
        incrementing the visit counts.
        """
        from aigames.mcts.mcts import GumbelActionSelector, GumbelHyperparameters, MCTSHyperparameters
        
        # Create a dummy MCTS object with the required attributes and methods
        class DummyMCTS:
            def __init__(self, action_selector):
                self.n_iters = 200  # Total number of iterations
                self.n_roots = 3  # Number of root nodes
                self.n_actions = 16  # Number of possible actions
                # Mask indicating legal actions for each root
                self.legal_actions_mask = torch.tensor([
                    [True for _ in range(self.n_actions)],
                    [True, True, False, False, True, False, True, False, True, False, True, False, True, False, True, False],
                    [True, False, True, True, False, True, False, True, False, True, False, True, False, True, False, True]
                ], dtype=torch.bool)
                self.legal_actions_mask = torch.cat([torch.zeros_like(self.legal_actions_mask), self.legal_actions_mask], dim=1).reshape(self.n_roots, -1, self.n_actions)
                # Random logits for each action at each root
                self.logits = (100*torch.arange(self.n_actions)).repeat(self.n_roots, 1)
                self.logits = torch.cat([torch.zeros_like(self.logits), self.logits], dim=1).reshape(self.n_roots, -1, self.n_actions)
                # Q-values function stub
                self.Q = lambda idx, cur_nodes: torch.zeros(idx.shape[0], self.n_actions)
                self.device = torch.device('cpu')
                # Initialize visit counts
                self.n = torch.zeros((self.n_roots, 2, 16), dtype=torch.int)
                self.hyperparams = MCTSHyperparameters(
                    dirichlet_alpha=1,
                    dirichlet_epsilon=0,
                    discount=0.99,
                    expand_simultaneous_fraction=1.0,
                    action_selector=action_selector
                )
        
        action_selector = GumbelActionSelector(GumbelHyperparameters(m=16))
        dummy_mcts = DummyMCTS(action_selector)
        
        # Initialize the GumbelActionSelector and perform MCTS initialization
        action_selector.on_mcts_init(dummy_mcts)
        action_selector.setup(dummy_mcts)
        
        torch.testing.assert_close(action_selector.cur_top_k, 2*torch.tensor([16, 8, 9], dtype=torch.int))
        
        # Iterate through a number of simulation steps
        for iteration in range(dummy_mcts.n_iters):
            # Generate actions for all roots
            root_indices = torch.arange(dummy_mcts.n_roots)
            current_nodes = torch.ones(dummy_mcts.n_roots, dtype=torch.int)
            actions = action_selector.get_actions(dummy_mcts, root_indices, current_nodes)
            
            # Ensure that the returned actions are within the valid range
            self.assertTrue(torch.all(actions >= 0) and torch.all(actions < dummy_mcts.n_actions),
                            "Action indices are out of valid range.")
            
            # Manually increment the visit counts for each root
            dummy_mcts.n[:, 0, 0] += 1
            dummy_mcts.n[torch.arange(dummy_mcts.n_roots), 1, actions] += 1
                    
        # Additional assertions can be added here to verify specific behaviors or properties
        # For example, checking if the top_actions are being updated correctly
        self.assertEqual(actions.shape, (dummy_mcts.n_roots,),
                         "The shape of the actions tensor is incorrect.")
