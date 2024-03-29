import enum
from dataclasses import dataclass
import torch
from aigames.game.game_multi import GameMulti
import json_fix


class UCBFormulaType(enum.Enum):
    Simple = 'simple'  # The simple UCB formula without a log term
    MuZeroLog = 'muzerolog'  # The UCB formula used in MuZero, with a log term

    # The UCB formula used in MuZero, with a log term, but visit all nodes from root at least once first
    MuZeroLogVisitAll = 'muzerolog_visitall'

    def __json__(self):
        return self.value


@dataclass(kw_only=True, slots=True)
class MCTSHyperparameters:
    c_puct: float = 1.0
    c_base_log_term: float = 19652.0  # The base for the log term in the UCB formula
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    discount: float = 1.0
    expand_simultaneous_fraction: float = 1.0  # The fraction of trees that need to be at a leaf to be expanded
    scaleQ: bool = True  # Whether to scale Q values to be between 0 and 1 my the min and max Q values in the tree
    ucb_formula: UCBFormulaType = UCBFormulaType.Simple


class MCTS:
    """
    Implementation of MCTS, trying to do simultaneous roll-outs of different nodes and use GPU as much as possible
    """

    def __init__(self, game: GameMulti, evaluator, hyperparams: MCTSHyperparameters, max_n_iters: int,
                 root_states: torch.Tensor, add_dirichlet_noise: bool = True):
        self.hyperparams = hyperparams
        self.evaluator = evaluator
        self.game = game
        self.n_iters = max_n_iters
        self.total_states = 2 + self.n_iters  # node 0 is dummy, 1 for the root, 1 for each iter
        self.device = root_states.device

        # The network's pi value for each root, state (outputs a policy size)
        n_roots = root_states.shape[0]
        n_actions = game.get_n_actions()
        state_shape = game.get_state_shape()
        n_players = game.get_n_players()
        n_stochastic_actions = game.get_n_stochastic_actions()

        self.state_shape = state_shape

        self.pi = torch.zeros(
            (n_roots, self.total_states, n_actions),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        self.legal_actions_mask = torch.zeros(
            (n_roots, self.total_states, n_actions),
            dtype=torch.bool,
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

        self.values = torch.zeros(
            (n_roots, self.total_states, n_players),
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

        self.is_env = torch.zeros(
            (n_roots, self.total_states,),
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

        self.maxQ = torch.zeros((n_roots, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.minQ = torch.zeros((n_roots, 1), dtype=torch.float32, device=self.device, requires_grad=False)

        self.root_idx = torch.arange(n_roots, dtype=torch.long, device=self.device, requires_grad=False)

        self.cur_nodes = torch.ones((n_roots,), dtype=torch.long, device=self.device,
                                    requires_grad=False)

        self.next_empty_nodes = 2 * torch.ones((n_roots,), dtype=torch.long,
                                               device=self.device, requires_grad=False)

        self.need_to_add_dirichlet_noise = torch.zeros((n_roots,), dtype=torch.bool,
                                                       device=self.device, requires_grad=False)

        self.only_one_action = torch.zeros((n_roots,), dtype=torch.bool, device=self.device, requires_grad=False)

        self.is_terminal[:, 1] = game.is_terminal(self.states[:, 1])
        self.pi[self.is_terminal[:, 1], 1, :] = 1.0 / n_actions
        self.n[self.is_terminal[:, 1], 1] = 1

        if add_dirichlet_noise:
            self.add_dirichlet_noise()

    def searchable_roots(self):
        out_of_bounds = (self.next_empty_nodes >= self.total_states) & (~self.is_leaf[self.root_idx, self.cur_nodes])
        done = self.n[:, 1].sum(dim=1) >= self.n_iters
        return (~out_of_bounds
                & ~done
                & ~self.only_one_action)

    def search(self):
        searchable_roots = self.searchable_roots()  # These are the roots we can do anything on
        leaves_or_terminal = self.is_leaf[self.root_idx, self.cur_nodes] | self.is_terminal[self.root_idx, self.cur_nodes]
        env = self.is_env[self.root_idx, self.cur_nodes]

        idx_env = self.root_idx[env & ~leaves_or_terminal & searchable_roots]
        cur_nodes_env = self.cur_nodes[env & ~leaves_or_terminal & searchable_roots]

        # Expand leaf nodes if enough trees are at a leaf (can set the fraction to 0 to always expand). 1e-3 is a tolerance for equality
        if leaves_or_terminal[searchable_roots].float().mean() + 1e-3 >= self.hyperparams.expand_simultaneous_fraction:
            # Expand leaves
            self.expand()

            # If we're at a terminal state, reset back to the root
            self.handle_terminal_states()

        # Advance env nodes, if searchable
        self.advance_to_next_states_from_env_states(idx_env, cur_nodes_env)

        # Refresh just in case we can be more efficient and do these all together now
        searchable_roots = self.searchable_roots()  # These are the roots we can do anything on
        leaves_or_terminal = self.is_leaf[self.root_idx, self.cur_nodes] | self.is_terminal[self.root_idx, self.cur_nodes]
        env = self.is_env[self.root_idx, self.cur_nodes]

        idx = self.root_idx[~leaves_or_terminal & ~env & searchable_roots]
        cur_nodes = self.cur_nodes[~leaves_or_terminal & ~env & searchable_roots]

        # Return if no trees to operate on
        if idx.shape[0] == 0:
            return

        # For the other nodes, choose best action and search down
        N = self.n[idx, cur_nodes]
        U = self.get_U(idx, cur_nodes, N)

        if self.hyperparams.scaleQ:
            allQ_for_max = (self.w[idx] - (self.n[idx] == 0).unsqueeze(-1).float()) / self.n[idx].unsqueeze(-1)  # w=0, n=0 => q=-inf
            allQ_for_min = (self.w[idx] + (self.n[idx] == 0).unsqueeze(-1).float()) / self.n[idx].unsqueeze(-1)  # w=0, n=0 => q=+inf

            maxQ = torch.maximum(allQ_for_max.amax(dim=(1, 2, 3)), self.values[idx, cur_nodes].amax(dim=(1,))).unsqueeze(-1)
            minQ = torch.minimum(allQ_for_min.amin(dim=(1, 2, 3)), self.values[idx, cur_nodes].amin(dim=(1,))).unsqueeze(-1)

            # Q replace unexplored edges with the given state's value
            Q = ((self.w[idx, cur_nodes, :, self.player_index[idx, cur_nodes]]
                  + (N == 0) * self.legal_actions_mask[idx, cur_nodes] * self.values[idx, cur_nodes])
                 / (N + (N == 0)))

            Q = (Q - minQ + 0.5*(maxQ == minQ))*self.legal_actions_mask[idx, cur_nodes] / (maxQ - minQ + (maxQ == minQ))
        else:
            Q = self.w[idx, cur_nodes, :, self.player_index[idx, cur_nodes]] / (N + (N == 0))

        next_actions = torch.argmax(Q + U, dim=1)

        self.advance_to_next_states(idx, cur_nodes, next_actions)

    def get_U(self, idx, nodes, N):
        if self.hyperparams.ucb_formula == UCBFormulaType.Simple:
            return self.hyperparams.c_puct * self.pi[idx, nodes] * torch.sqrt(1 + N.sum(dim=1, keepdim=True)) / (1 + N)
        elif self.hyperparams.ucb_formula == UCBFormulaType.MuZeroLog:
            return (
                self.pi[idx, nodes] * torch.sqrt(1 + N.sum(dim=1, keepdim=True)) / (1 + N)
                * (self.hyperparams.c_puct
                   + torch.log((1 + N.sum(dim=1, keepdim=True) + self.hyperparams.c_base_log_term) / self.hyperparams.c_base_log_term))
            )
        elif self.hyperparams.ucb_formula == UCBFormulaType.MuZeroLogVisitAll:
            return (
                self.pi[idx, nodes] * torch.sqrt(1 + N.sum(dim=1, keepdim=True)) / (1 + N)
                * (self.hyperparams.c_puct
                   + torch.log((1 + N.sum(dim=1, keepdim=True) + self.hyperparams.c_base_log_term) / self.hyperparams.c_base_log_term))
                + (1./(1 - ((N == 0.) & (nodes == 1).unsqueeze(-1) & self.legal_actions_mask[idx, nodes]).float()) - 1.)  # inf if N == 0 and nodes == 1
            )

    def search_for_n_iters(self, n_iters):
        searchable = self.searchable_roots()

        while searchable.any().item() and self.n[searchable, 1].sum(dim=1).min() < n_iters:
            self.search()
            searchable = self.searchable_roots()

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

        # Advance to the already filled-in nodes
        self.cur_nodes[idx[mask]] = next_idx[mask]

        # If the next idx is zero then we need to fill it in
        self.fill_in_next_states(idx[~mask], nodes[~mask], next_actions[~mask])

    def fill_in_next_states(self, idx, cur_nodes, next_actions):
        # Get the next states
        next_states, rewards, is_env, is_terminal = self.game.get_next_states(self.states[idx, cur_nodes], next_actions)

        next_nodes = self.next_empty_nodes[idx]
        self.states[idx, next_nodes] = next_states
        self.rewards[idx, next_nodes] = rewards
        self.is_terminal[idx, next_nodes] = is_terminal
        self.next_idx[idx, cur_nodes, next_actions] = next_nodes
        self.parent_nodes[idx, next_nodes] = cur_nodes.clone()
        self.is_env[idx, next_nodes] = is_env
        self.actions[idx, next_nodes] = next_actions
        self.cur_nodes[idx] = next_nodes

        # Set the action
        self.next_empty_nodes[idx] += 1

    def advance_to_next_states_from_env_states(self, idx, nodes):
        # Now get the next player state for the env states
        next_player_states, env_action_idx, is_terminal = self.game.get_next_states_from_env(self.states[idx, nodes])

        next_node_idx = self.next_idx_env[idx, nodes, env_action_idx]

        # If the next node idx is zero then we need to fill it in
        empty_mask = (next_node_idx == 0)
        empty_idx = idx[empty_mask]
        empty_nodes = nodes[empty_mask]

        next_nodes = self.next_empty_nodes[empty_idx]
        self.states[empty_idx, next_nodes] = next_player_states[empty_mask]
        self.is_terminal[empty_idx, next_nodes] = is_terminal[empty_mask]
        self.next_idx_env[empty_idx, empty_nodes, env_action_idx[empty_mask]] = next_nodes
        self.cur_nodes[empty_idx] = next_nodes
        self.parent_nodes[empty_idx, next_nodes] = empty_nodes
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
        network_result = self.evaluator.evaluate(states)
        self.pi[idx, nodes], values = network_result[0], network_result[1]
        self.values[idx, nodes] = values
        self.legal_actions_mask[idx, nodes] = self.game.get_legal_action_masks(states)
        self.pi[idx, nodes] *= self.legal_actions_mask[idx, nodes]
        # re-normalize pi
        self.pi[idx, nodes] /= self.pi[idx, nodes].sum(dim=1, keepdim=True)

        # For any roots, if there is only one valid action, set the flag; prevents searching for efficiency
        self.set_only_one_action(idx, nodes, self.legal_actions_mask[idx, nodes])

        parents = self.parent_nodes[idx, nodes]
        has_parent = parents.to(torch.bool)
        nodesp = nodes[has_parent]
        idxp = idx[has_parent]

        # back up values
        if nodesp.shape[0] > 0:
            self.backup(idxp, self.parent_nodes[idxp, nodesp], self.actions[idxp, nodesp],
                        self.rewards[idxp, nodesp] + (self.hyperparams.discount * values))

        # no longer leaves
        self.is_leaf[idx, nodes] = False

        # Add dirichlet noise if needed
        if self.need_to_add_dirichlet_noise[idx].any():
            self._add_dirichlet_noise(idx[self.need_to_add_dirichlet_noise[idx]])
            self.need_to_add_dirichlet_noise[idx] = False

        # reset back to root and increment count
        self.cur_nodes[is_leaf_mask] = 1

    def set_only_one_action(self, idx, nodes, legal_actions_mask):
        roots = (nodes == 1)
        self.only_one_action[idx[roots]] = (legal_actions_mask[roots].sum(dim=1) == 1)
        # Give one visit in n to the only legal action
        only_one_action = roots & self.only_one_action[idx]
        self.n[
            idx[only_one_action], nodes[only_one_action], legal_actions_mask[only_one_action].to(int).argmax(dim=1)] = 1

    def backup(self, idx, nodes, actions, values):
        if nodes.shape[0] == 0:
            return

        self.w[idx, nodes, actions] += values
        self.n[idx, nodes, actions] += 1

        parents = self.parent_nodes[idx, nodes]
        actions = self.actions[idx, nodes]
        q = self.rewards[idx, nodes] + self.hyperparams.discount * values
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

    def get_next_mcts(self, new_roots: torch.Tensor, add_dirichlet_noise=True):
        """
        Preserve as much of the current tree as possible with a new tree whose roots are new_states

        :param new_roots: The new states to start the new tree from
        """
        return
        # TODO : fix this for new style
        mcts = MCTS(self.game, self.evaluator, self.hyperparams, new_roots)
        mcts.next_empty_nodes[:] = 1  # For convenience, we're just going to overwrite the root

        # Maybe you could try to do these in parallel? but much simpler to write by doing each root one by one
        for root in self.root_idx:
            state_match = (self.states[root] == new_roots[root])
            for _ in self.state_shape:
                state_match = state_match.all(dim=1)

            node_idx = state_match.nonzero().flatten()
            if node_idx.shape[0] == 0:
                mcts.next_empty_nodes[root] = 2  # This state was not visited in existing tree, just keep the root we set in the constructor
                continue

            if node_idx.shape[0] > 1:
                best = self.n[root, node_idx].sum(dim=1).argmax()
                node_idx = node_idx[best].item()
            else:
                node_idx = node_idx.item()

            player_nodes = [(0, None, None, node_idx)]  # Format is: parent_node, env_parent_node (or None), env_parent_action (or None), node_idx
            env_nodes = []  # Format is parent_node, node_idx, action

            while len(player_nodes) > 0 or len(env_nodes) > 0:
                # Handle all the player_nodes first
                if len(player_nodes) > 0:
                    new_parent, env_parent, env_action, node = player_nodes.pop()
                    new_node = mcts.next_empty_nodes[root].clone()

                    mcts.states[root, new_node] = self.states[root, node]
                    mcts.rewards[root, new_node] = self.rewards[root, node]
                    mcts.is_terminal[root, new_node] = self.is_terminal[root, node]
                    mcts.parent_nodes[root, new_node] = new_parent
                    mcts.pi[root, new_node] = self.pi[root, node]
                    mcts.n[root, new_node] = self.n[root, node]
                    mcts.w[root, new_node] = self.w[root, node]
                    mcts.player_index[root, new_node] = self.player_index[root, node]
                    mcts.actions[root, new_node] = self.actions[root, node]
                    mcts.is_leaf[root, new_node] = self.is_leaf[root, node]
                    mcts.env_is_next[root, new_node] = self.env_is_next[root, node]

                    if env_parent is not None:
                        mcts.next_idx_env[root, env_parent, env_action] = new_node
                    elif new_parent > 0:
                        mcts.next_idx[root, new_parent, self.actions[root, node]] = new_node

                    # Enqueue the children
                    for action, next_idx in enumerate(self.next_idx[root, node]):
                        if next_idx != 0:
                            if self.env_is_next[root, node, action]:
                                env_nodes.append((new_node, next_idx, action))
                            else:
                                player_nodes.append((new_node, None, None, next_idx))

                    mcts.next_empty_nodes[root] += 1
                else:
                    new_parent, env_node, action = env_nodes.pop()
                    new_env_node = mcts.next_empty_nodes_env[root].clone()
                    mcts.env_states[root, new_env_node] = self.env_states[root, env_node]
                    mcts.env_state_rewards[root, new_env_node] = self.env_state_rewards[root, env_node]
                    mcts.next_idx[root, new_parent, action] = new_env_node

                    # Enqueue the children
                    for action, next_idx in enumerate(self.next_idx_env[root, env_node]):
                        if next_idx != 0:
                            player_nodes.append((new_parent, new_env_node, action, next_idx))

                    mcts.next_empty_nodes_env[root] += 1

        mcts.set_only_one_action(mcts.root_idx, torch.ones((mcts.root_idx.shape[0],), dtype=torch.long, device=mcts.device),
                                 (mcts.pi[mcts.root_idx, 1] > 0.0))

        assert ((mcts.parent_nodes[:, 1] != 1).all())

        return mcts
