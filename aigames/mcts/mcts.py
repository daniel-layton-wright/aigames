from typing import Type
import torch
from aigames.game.game_multi import GameMulti


class MCTSHyperparameters:
    __slots__ = ['n_roots', 'n_iters', 'c_puct', 'dirichlet_alpha', 'dirichlet_epsilon']

    def __init__(self):
        self.n_roots = 10
        self.n_iters = 1200
        self.c_puct = 1.0
        self.dirichlet_alpha = 0.3
        self.dirichlet_epsilon = 0.25


class MCTS:
    """
    Implementation of MCTS, trying to do simultaneous roll-outs of different nodes and use GPU as much as possible
    """

    def __init__(self, game: GameMulti, evaluator, hyperparams, root_states: torch.Tensor):
        self.hyperparams = hyperparams
        self.evaluator = evaluator
        self.game = game
        self.total_states = 2 + self.hyperparams.n_iters
        self.device = root_states.device

        # The network's pi value for each root, state (outputs a policy size)
        n_roots = root_states.shape[0]
        n_actions = game.get_n_actions()
        state_shape = game.get_state_shape()
        n_players = game.get_n_players()
        n_stochastic_actions = game.get_n_stochastic_actions()

        self.pi = torch.zeros(
            (n_roots, self.total_states, n_actions),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        # The number of visits for each child of each root, state
        self.n = torch.zeros(
            (n_roots, self.total_states, n_actions),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        self.player_index = torch.zeros(
            (n_roots, self.total_states,),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )

        # The total values that have been backed-up the tree for each child of each root, state
        self.w = torch.zeros(
            (n_roots, self.total_states, n_actions, n_players),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        self.states = torch.zeros(
            (n_roots, self.total_states, *state_shape),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        self.states[:, 1] = root_states

        self.rewards = torch.zeros(
            (n_roots, self.total_states, n_players),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        self.env_state_rewards = torch.zeros(
            (n_roots, self.total_states, n_players),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        self.env_states = torch.zeros(
            (n_roots, self.total_states, *state_shape),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        # The action from the parent that got to this node (used for backing up values)
        self.actions = torch.zeros(
            (n_roots, self.total_states,),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )

        self.next_idx = torch.zeros(
            (n_roots, self.total_states, n_actions),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )

        self.next_idx_env = torch.zeros(
            (n_roots, self.total_states, n_stochastic_actions),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )

        self.env_is_next = torch.zeros(
            (n_roots, self.total_states, n_actions),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False
        )

        self.is_leaf = torch.ones(
            (n_roots, self.total_states,),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False
        )

        self.is_terminal = torch.zeros(
            (n_roots, self.total_states,),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False
        )

        self.parent_nodes = torch.zeros(
            (n_roots, self.total_states,),
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )

        self.root_idx = torch.arange(n_roots, dtype=torch.long, device=self.device)

        self.cur_nodes = torch.ones((n_roots,), dtype=torch.long, device=self.device,
                                    requires_grad=False)

        self.next_actions = torch.zeros((n_roots,), dtype=torch.long, device=self.device,
                                        requires_grad=False)

        self.next_empty_nodes = 2 * torch.ones((n_roots,), dtype=torch.long,
                                               device=self.device, requires_grad=False)

        self.next_empty_nodes_env = torch.ones((n_roots,), dtype=torch.long,
                                                device=self.device, requires_grad=False)

        self.need_to_add_dirichlet_noise = torch.zeros((n_roots,), dtype=torch.bool,
                                                       device=self.device, requires_grad=False)

        self.only_one_action = torch.zeros((n_roots,), dtype=torch.bool, device=self.device, requires_grad=False)

        self.is_terminal[:, 1] = game.is_terminal(self.states[:, 1])
        self.pi[self.is_terminal[:, 1], 1, :] = 1.0 / n_actions
        self.n[self.is_terminal[:, 1], 1] = 1

    def searchable_roots(self):
        out_of_bounds = (self.next_empty_nodes >= self.total_states) | (self.next_empty_nodes_env > self.total_states)
        return (~out_of_bounds
                & ~self.only_one_action)

    def search(self):
        not_leaves_or_terminal = (~self.is_leaf[self.root_idx, self.cur_nodes]
                               & ~self.is_terminal[self.root_idx, self.cur_nodes])
        cur_searchable = not_leaves_or_terminal & self.searchable_roots()

        cur_nodes = self.cur_nodes[cur_searchable]
        idx = self.root_idx[cur_searchable]

        # If we're at a terminal state, reset back to the root
        self.handle_terminal_states()

        # Expand leaf nodes
        self.expand()

        # For the other nodes, choose best action and search down
        N = self.n[idx, cur_nodes]
        U = (self.hyperparams.c_puct * self.pi[idx, cur_nodes] *
             torch.sqrt(1 + N.sum(dim=1, keepdim=True)) / (1 + N))

        Q = self.w[idx, cur_nodes, :, self.player_index[idx, cur_nodes]] / (N + (N == 0))  # If N == 0 -> we make it 1 to avoid dividing by zero

        next_actions = torch.argmax(Q + U, dim=1)

        self.advance_to_next_states(idx, cur_nodes, next_actions)

    def search_for_n_iters(self, n_iters):
        while any(self.searchable_roots()) > 0 and self.n[self.searchable_roots()].sum(dim=2).min() < n_iters:
            self.search()

    def handle_terminal_states(self):
        """
        If any states are terminal, do a backup, then reset those back to the root
        """
        terminal_mask = self.is_terminal[self.root_idx, self.cur_nodes]
        idx = self.root_idx[terminal_mask]
        nodes = self.cur_nodes[terminal_mask]

        self.backup(idx, self.parent_nodes[idx, nodes], self.actions[idx, nodes],
                    self.rewards[idx, nodes])

        self.cur_nodes[terminal_mask] = 1

    def advance_to_next_states(self, idx, nodes, next_actions):
        next_idx = self.next_idx[idx, nodes, next_actions]

        # If the next idx is not zero then it has already been filled in
        mask = (next_idx != 0)

        # If the next idx is not an env state, we can go right to it
        is_env = self.env_is_next[idx[mask], nodes[mask], next_actions[mask]]
        self.cur_nodes[idx[mask][~is_env]] = next_idx[mask][~is_env]

        # If the next idx is zero then we need to fill it in
        self.fill_in_next_states(idx[~mask], nodes[~mask], next_actions[~mask])

        # Now we have a valid next_idx for everybody
        # If the next idx is an env state, we need to get the next player state
        is_env = self.env_is_next[idx, nodes, next_actions]
        next_idx_env = self.next_idx[idx, nodes, next_actions][is_env]
        self.advance_to_player_states_from_env_states(idx[is_env], next_idx_env, nodes[is_env])

        # Set the action
        self.actions[idx, self.cur_nodes[idx]] = next_actions

    def fill_in_next_states(self, idx, cur_nodes, next_actions):
        # Get the next states
        next_states, rewards, is_env, is_terminal = self.game.get_next_states(self.states[idx, cur_nodes], next_actions)

        self.env_is_next[idx, cur_nodes, next_actions] = is_env

        # For non-env next states, put them in the states variable
        non_env = torch.logical_not(is_env)
        idx_non_env = idx[non_env]
        next_nodes = self.next_empty_nodes[idx_non_env]
        self.states[idx_non_env, next_nodes] = next_states[non_env]
        self.rewards[idx_non_env, next_nodes] = rewards[non_env]
        self.is_terminal[idx_non_env, next_nodes] = is_terminal[non_env]
        self.next_idx[idx_non_env, cur_nodes[non_env], next_actions[non_env]] = next_nodes
        self.parent_nodes[idx_non_env, next_nodes] = cur_nodes[non_env].clone()
        self.cur_nodes[idx_non_env] = next_nodes
        self.next_empty_nodes[idx_non_env] += 1

        # For env states store them in the env_states variable
        idx_env = idx[is_env]
        next_nodes_env = self.next_empty_nodes_env[idx_env]
        self.env_states[idx_env, next_nodes_env] = next_states[is_env]
        self.env_state_rewards[idx_env, next_nodes_env] = rewards[is_env]
        self.next_idx[idx_env, cur_nodes[is_env], next_actions[is_env]] = next_nodes_env

        self.next_empty_nodes_env[idx_env] += 1

    def advance_to_player_states_from_env_states(self, idx, env_nodes, parent_nodes):
        # Now get the next player state for the env states
        next_player_states, env_action_idx, is_terminal = self.game.get_next_states_from_env(self.env_states[idx, env_nodes])

        next_node_idx = self.next_idx_env[idx, env_nodes, env_action_idx]

        # If the next node idx is zero then we need to fill it in
        empty_mask = (next_node_idx == 0)
        empty_idx = idx[empty_mask]
        empty_nodes = env_nodes[empty_mask]
        next_nodes = self.next_empty_nodes[empty_idx]
        self.states[empty_idx, next_nodes] = next_player_states[empty_mask]
        self.rewards[empty_idx, next_nodes] = self.env_state_rewards[empty_idx, empty_nodes]
        self.is_terminal[empty_idx, next_nodes] = is_terminal[empty_mask]
        self.next_idx_env[empty_idx, empty_nodes, env_action_idx[empty_mask]] = next_nodes
        self.cur_nodes[empty_idx] = next_nodes
        self.parent_nodes[empty_idx, next_nodes] = parent_nodes[empty_mask].clone()
        self.next_empty_nodes[empty_idx] += 1

        # If the next node idx is not zero then we've already been there, just advance to it
        self.cur_nodes[idx[~empty_mask]] = next_node_idx[~empty_mask]

    def expand(self):
        # Get the states, rewards, evaluate network and store results
        is_leaf_mask = self.is_leaf[self.root_idx, self.cur_nodes] & ~self.is_terminal[self.root_idx, self.cur_nodes]
        nodes = self.cur_nodes[is_leaf_mask]
        idx = self.root_idx[is_leaf_mask]

        if nodes.shape[0] == 0:
            return

        states = self.states[idx, nodes]
        self.pi[idx, nodes], values = self.evaluator.evaluate(states)
        legal_actions_mask = self.game.get_legal_action_masks(states)
        self.pi[idx, nodes] *= legal_actions_mask
        # re-normalize pi
        self.pi[idx, nodes] /= self.pi[idx, nodes].sum(dim=1, keepdim=True)

        # For any roots, if there is only one valid action, set the flag; prevents searching for efficiency
        roots = (nodes == 1)
        self.only_one_action[idx[roots]] = (legal_actions_mask[roots].sum(dim=1) == 1)
        # Give one visit in n to the only legal action
        only_one_action = roots & self.only_one_action[idx]
        self.n[idx[only_one_action], nodes[only_one_action], legal_actions_mask[only_one_action].to(int).argmax(dim=1)] = 1

        parents = self.parent_nodes[idx, nodes]
        has_parent = parents.to(torch.bool)
        nodesp = nodes[has_parent]
        idxp = idx[has_parent]

        # back up values
        if nodesp.shape[0] > 0:
            self.backup(idxp, self.parent_nodes[idxp, nodesp], self.actions[idxp, nodesp],
                        values + self.rewards[idxp, nodesp])

        # no longer leaves
        self.is_leaf[idx, nodes] = False

        # Add dirichlet noise if needed
        if self.need_to_add_dirichlet_noise[idx].any():
            self._add_dirichlet_noise(idx[self.need_to_add_dirichlet_noise[idx]])
            self.need_to_add_dirichlet_noise[idx] = False

        # reset back to root and increment count
        self.cur_nodes[is_leaf_mask] = 1

    def backup(self, idx, nodes, actions, values):
        if nodes.shape[0] == 0:
            return

        self.w[idx, nodes, actions] += values
        self.n[idx, nodes, actions] += 1

        parents = self.parent_nodes[idx, nodes]
        actions = self.actions[idx, nodes]
        q = values + self.rewards[idx, nodes]
        mask = (parents != 0)
        self.backup(idx[mask], parents[mask], actions[mask], q[mask])

    def add_dirichlet_noise(self):
        expanded_roots = ~self.is_leaf[self.root_idx, 1]
        self._add_dirichlet_noise(self.root_idx[expanded_roots])

        self.need_to_add_dirichlet_noise[~expanded_roots] = True

    def _add_dirichlet_noise(self, idx):
        # Add dirichlet noise to the root nodes
        mask = (self.pi[idx, 1] > 0)

        dirichlet = torch.distributions.dirichlet.Dirichlet(
            torch.full((self.game.get_n_actions(),), self.hyperparams.dirichlet_alpha, dtype=torch.float32)
        ).sample(torch.Size((len(idx),))).to(self.device)

        # Apply mask and renormalize
        dirichlet *= mask
        dirichlet /= dirichlet.sum(dim=1, keepdim=True)

        self.pi[idx, 1] = ((1 - self.hyperparams.dirichlet_epsilon) * self.pi[idx, 1]
                           + self.hyperparams.dirichlet_epsilon * dirichlet)
