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


class ActionSelector:
    def get_actions(self, mcts, idx, cur_nodes):
        raise NotImplementedError()
    
    def get_final_actions_and_pi(self, mcts, tau):
        raise NotImplementedError()
    
    def on_mcts_init(self, mcts):
        pass


@dataclass(kw_only=True, slots=True)
class UCBHyperparameters:
    c_puct: float = 1.0
    c_base_log_term: float = 19652.0  # The base for the log term in the UCB formula
    scaleQ: bool = True  # Whether to scale Q values to be between 0 and 1 my the min and max Q values in the tree
    ucb_formula: UCBFormulaType = UCBFormulaType.Simple


class UCBActionSelector(ActionSelector):
    def __init__(self, hyperparams: UCBHyperparameters):
        self.hyperparams = hyperparams

    def Q(self, mcts, idx, cur_nodes):
        return mcts.Q(idx, cur_nodes, scale_Q=self.hyperparams.scaleQ)

    def U(self, mcts, idx, nodes):
        N = mcts.n[idx, nodes]
        if self.hyperparams.ucb_formula == UCBFormulaType.Simple:
            return self.hyperparams.c_puct * mcts.pi[idx, nodes] * torch.sqrt(1 + N.sum(dim=1, keepdim=True)) / (1 + N)
        elif self.hyperparams.ucb_formula == UCBFormulaType.MuZeroLog:
            return (
                mcts.pi[idx, nodes] * torch.sqrt(1 + N.sum(dim=1, keepdim=True)) / (1 + N)
                * (self.hyperparams.c_puct
                   + torch.log((1 + N.sum(dim=1, keepdim=True) + self.hyperparams.c_base_log_term) / self.hyperparams.c_base_log_term))
            )
        elif self.hyperparams.ucb_formula == UCBFormulaType.MuZeroLogVisitAll:
            return (
                mcts.pi[idx, nodes] * torch.sqrt(1 + N.sum(dim=1, keepdim=True)) / (1 + N)
                * (self.hyperparams.c_puct
                   + torch.log((1 + N.sum(dim=1, keepdim=True) + self.hyperparams.c_base_log_term) / self.hyperparams.c_base_log_term))
                + (1./(1 - ((N == 0.) & (nodes == 1).unsqueeze(-1) & mcts.legal_actions_mask[idx, nodes]).float()) - 1.)  # inf if N == 0 and nodes == 1
            )

    def get_actions(self, mcts, idx, cur_nodes):
        Q = self.Q(mcts, idx, cur_nodes)
        U = self.U(mcts, idx, cur_nodes)

        return (Q + U).argmax(dim=1)
    
    def get_final_actions_and_pi(self, mcts, tau):
        n_parallel_games = mcts.n_roots
        pi = torch.zeros((n_parallel_games, mcts.n_actions), device=mcts.device, dtype=torch.float32)
        actions = torch.zeros(n_parallel_games, dtype=torch.long, device=mcts.device)

        if isinstance(tau, float):
            tau = torch.tensor([[tau]], device=mcts.device, dtype=torch.float32).repeat(n_parallel_games, 1)        

        if mcts.n[:, 0, 0].amax() > 1:
            action_distribution = mcts.n[:, 1]
        else:
            mcts.expand()
            action_distribution = mcts.pi[:, 1]

        non_zero = (tau > 0).flatten()
        pi[non_zero] = action_distribution[non_zero] ** (1. / tau[non_zero])
        pi[non_zero] /= pi[non_zero].sum(dim=1, keepdim=True)
        # Choose the action according to the distribution pi
        actions[non_zero] = torch.multinomial(pi[non_zero], num_samples=1).flatten()

        pi[torch.arange(n_parallel_games, device=non_zero.device)[~non_zero], action_distribution[~non_zero].argmax(dim=1)] = 1
        actions[~non_zero] = action_distribution[~non_zero].argmax(dim=1).flatten()

        return actions, pi


@dataclass(kw_only=True, slots=True)
class GumbelHyperparameters:
    c_visit: float = 50.0
    c_scale: float = 1.0
    m: int = 8  # max number of actions to sample without replacement in first round


class GumbelActionSelector(ActionSelector):
    """
    Based on the paper: POLICY IMPROVEMENT BY PLANNING WITH GUMBEL
    https://openreview.net/pdf?id=bERaNdoegnO
    
    Relevant algorithm:
    Algorithm 2 Sequential Halving with Gumbel
    Require: k: number of actions.
    Require: m ≤ k: number of actions sampled without replacement.
    Require: n: number of simulations.
    Require: logits ∈ Rk: predictor logits from a policy network π.
    Sample k Gumbel variables:
    (g ∈ Rk) ~ Gumbel(0)
    Find m actions with the highest g(a) + logits(a):
    Atopm = argtop(g + logits, m)
    Use Sequential Halving with n simulations to identify the best action from the Atopm actions,
    by comparing g(a) + logits(a) + sigma(q_hat(a)).
    An+1 = arg maxa∈Remaining (g(a) + logits(a) + sigma(q_hat(a)))
    return An+1
    
    sigma(q_hat(a)) = (c_visit + max N(b)) * c_scale * q_hat(a), 
    """
    def __init__(self, hyperparams: GumbelHyperparameters):
        self.hyperparams = hyperparams
        self.gumbel = torch.zeros(0)  # will be filled in later
        self.starting_top_k = torch.zeros(0)
        self.cur_top_k = torch.zeros(0)
        self.last_update_at = torch.zeros(0)
        self.next_update_at = torch.zeros(0)
        self.top_actions = torch.zeros(0)
        self.k_schedule = []
        self.n_schedule = []
        self._need_setup = True
        
        
    def sigma(self, N, Q):
        return (self.hyperparams.c_visit + N.amax(dim=1, keepdim=True)) * self.hyperparams.c_scale * Q
    
    def on_mcts_init(self, mcts):
        self._need_setup = True

    def setup(self, mcts):
        n = mcts.n_iters
        
        self.gumbel = torch.distributions.gumbel.Gumbel(0, 1).sample((mcts.n_roots, mcts.n_actions)).to(mcts.device)
        
        n_legal_actions_at_root = mcts.legal_actions_mask[:, 1].sum(dim=1)
        self.cur_top_k = 2*torch.minimum(n_legal_actions_at_root, torch.tensor(self.hyperparams.m, device=mcts.device))
        self.last_update_at = torch.zeros(mcts.n_roots, dtype=torch.int, device=mcts.device)
        self.next_update_at = torch.zeros(mcts.n_roots, dtype=torch.int, device=mcts.device) # (torch.floor(n / (torch.log2(self.cur_top_k) * self.cur_top_k)) * self.cur_top_k).to(torch.int)
        self.cur_top_k = self.cur_top_k.to(torch.int)
        self.starting_top_k = self.cur_top_k / 2
        max_k = int(self.starting_top_k.amax())
        self.top_actions = torch.zeros(mcts.n_roots, max_k, dtype=torch.long, device=mcts.device)
        
        self._need_setup = False
        
    def get_actions(self, mcts, idx, cur_nodes):
        if self._need_setup:
            self.setup(mcts)
        
        roots = (cur_nodes == 1)
        actions = torch.zeros((idx.shape[0],), dtype=torch.long, device=mcts.device)
        actions[roots] = self.get_actions_roots(mcts, idx[roots], cur_nodes[roots])
        actions[~roots] = self.get_actions_non_roots(mcts, idx[~roots], cur_nodes[~roots])
        return actions

    def get_actions_non_roots(self, mcts, idx, cur_nodes):
        """
        arg max pi'(a) - [ N(a) / [ 1 + ∑b N(b) ] ]
        where pi' = softmax(logits + sigma(completedQ))
        """
        pi_prime = torch.softmax(mcts.logits[idx, cur_nodes] + self.sigma(mcts.n[idx, cur_nodes], mcts.Q(idx, cur_nodes)), dim=1)
        objective = pi_prime - (mcts.n[idx, cur_nodes] / (1 + mcts.n[idx, cur_nodes].sum(dim=1, keepdim=True)))
        return objective.argmax(dim=1)
        

    def get_actions_roots(self, mcts, idx, cur_nodes):
        if len(idx) == 0:
            return self.top_actions[idx, torch.zeros_like(idx)]
        
        total_n = mcts.n_iters
        
        N = mcts.n[idx, 0, 0]  # current iteration count
        need_to_update = idx[N == self.next_update_at[idx]]
        update_nodes = cur_nodes[N == self.next_update_at[idx]]
        
        self.cur_top_k[need_to_update] = torch.maximum(self.cur_top_k[need_to_update] // 2, torch.tensor(2, device=self.cur_top_k.device))
        self.last_update_at[need_to_update] = mcts.n[need_to_update, 0, 0]
        self.next_update_at[need_to_update] = (self.last_update_at[need_to_update] + 
                                               (torch.floor(total_n / 
                                                            (torch.log2(self.starting_top_k[need_to_update].float()) * self.cur_top_k[need_to_update].float()))
                                                * self.cur_top_k[need_to_update]).to(torch.int))
        
        max_k = int(self.cur_top_k[idx].amax())
        self.top_actions[need_to_update, :max_k] = (mcts.logits[need_to_update, update_nodes] + self.gumbel[need_to_update] 
                                                    + self.sigma(mcts.n[need_to_update, update_nodes], mcts.Q(need_to_update, update_nodes))
                                                    ).topk(dim=1, k=max_k, sorted=True).indices
        
        top_action_index = (N - self.last_update_at[idx]) % self.cur_top_k[idx]
        return self.top_actions[idx, top_action_index]
    
    def get_final_actions_and_pi(self, mcts, tau):
        """
        The final action is the one with the highest logit + gumbel + sigma(Q)
        The pi to be used for training is the softmax of the logits + sigma(Q)
        """
        idx = mcts.root_idx
        nodes = 1
        actions = (mcts.logits[idx, nodes] + self.sigma(mcts.n[idx, nodes], mcts.Q(idx, nodes))).argmax(dim=1)
        pi = torch.softmax(mcts.logits[idx, nodes] + self.sigma(mcts.n[idx, nodes], mcts.Q(idx, nodes)), dim=1)
        return actions, pi


@dataclass(kw_only=True, slots=True)
class MCTSHyperparameters:
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    discount: float = 1.0
    expand_simultaneous_fraction: float = 1.0  # The fraction of trees that need to be at a leaf to be expanded
    action_selector: ActionSelector = UCBActionSelector(UCBHyperparameters())


class MCTS:
    """
    Implementation of MCTS, trying to do simultaneous roll-outs of different nodes and use GPU as much as possible
    """

    def __init__(self, game: GameMulti, evaluator, hyperparams: MCTSHyperparameters, 
                 max_n_iters: int, root_states: torch.Tensor, add_dirichlet_noise: bool = True):
        self.hyperparams = hyperparams
        self.evaluator = evaluator
        self.game = game
        self.n_iters = max_n_iters
        self.total_states = 2 + self.n_iters  # node 0 is dummy, 1 for the root, 1 for each iter
        self.device = root_states.device

        # The network's pi value for each root, state (outputs a policy size)
        n_roots = root_states.shape[0]
        self.n_roots = n_roots
        self.n_actions = game.get_n_actions()
        state_shape = game.get_state_shape()
        n_players = game.get_n_players()
        self.n_stochastic_actions = game.get_n_stochastic_actions()

        self.n_actions_max = max(self.n_actions, self.n_stochastic_actions)

        self.state_shape = state_shape

        self.pi = torch.zeros(
            (n_roots, self.total_states, self.n_actions),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        
        self.logits = torch.zeros(
            (n_roots, self.total_states, self.n_actions),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        self.legal_actions_mask = torch.zeros(
            (n_roots, self.total_states, self.n_actions_max),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False
        )

        # The number of visits for each child of each root, state
        self.n = torch.zeros(
            (n_roots, self.total_states, self.n_actions),
            dtype=torch.int,
            device=self.device,
            requires_grad=False
        )

        self.player_index = torch.zeros(
            (n_roots, self.total_states,),
            dtype=torch.int,
            device=self.device,
            requires_grad=False
        )

        # The total values that have been backed-up the tree for each child of each root, state
        self.w = torch.zeros(
            (n_roots, self.total_states, self.n_actions, n_players),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        self.states = torch.zeros(
            (n_roots, self.total_states, *state_shape),
            dtype=root_states.dtype,
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
            (n_roots, self.total_states, self.n_actions_max),
            dtype=torch.int,
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
            dtype=torch.int,
            device=self.device,
            requires_grad=False
        )

        self.maxQ = torch.nan * torch.ones((n_roots,), dtype=torch.float32, device=self.device, requires_grad=False)
        self.minQ = torch.nan * torch.ones((n_roots,), dtype=torch.float32, device=self.device, requires_grad=False)

        self.root_idx = torch.arange(n_roots, dtype=torch.int, device=self.device, requires_grad=False)
        self.cur_nodes = torch.ones((n_roots,), dtype=torch.int, device=self.device, requires_grad=False)

        # Initialize a list of empty nodes for each root, which should be arange(2, self.total_nodes) for each root
        # If you use the 'subtree persistence' feature, this won't necessarily be arange, because the nodes down an
        # untaken path will be able to be overwritten and re-used and the existing nodes will be kept in their place
        # (to be faster)
        self.empty_nodes = torch.arange(2, self.total_states + 1, device=self.device, requires_grad=False, dtype=torch.int,
                                        ).unsqueeze(0).expand(n_roots, -1)

        # This is the index in empty_nodes of where we should put the next node
        # So: if next_empty_node_pointers[0] is 3, then the next node for root 0 should be put in empty_nodes[0, 3]
        self.next_empty_node_pointers = torch.zeros((n_roots,), dtype=torch.int,
                                               device=self.device, requires_grad=False)

        self.need_to_add_dirichlet_noise = torch.zeros((n_roots,), dtype=torch.bool,
                                                       device=self.device, requires_grad=False)

        self.only_one_action = torch.zeros((n_roots,), dtype=torch.bool, device=self.device, requires_grad=False)
        self.handle_terminal_roots()
        
        self.action_selector.on_mcts_init(self)

        if add_dirichlet_noise:
            self.add_dirichlet_noise()
            
    @property
    def action_selector(self):
        return self.hyperparams.action_selector

    @property
    def r(self):
        return self.root_idx, 1

    @property
    def next_empty_nodes(self):
        """
        This property looks up the next empty node based on the next_empty_node_pointers and the empty_nodes
        """
        return self.empty_nodes[self.root_idx, self.next_empty_node_pointers]

    def searchable_roots(self):
        out_of_bounds = (self.next_empty_nodes >= self.total_states) & (~self.is_leaf[self.root_idx, self.cur_nodes])
        done = self.n[:, 0, 0] >= self.n_iters
        return (~out_of_bounds
                & ~done
                & ~self.only_one_action)

    def handle_terminal_roots(self):
        self.is_terminal[self.r] = self.game.is_terminal(self.states[self.r])
        term_roots = self.is_terminal[self.r]
        self.pi[term_roots, 1, :] = 1.0 / self.n_actions
        self.n[term_roots, 1] = 1

    def search(self):
        searchable_roots = self.searchable_roots()  # These are the roots we can do anything on
        leaves_or_terminal = self.is_leaf[self.root_idx, self.cur_nodes] | self.is_terminal[self.root_idx, self.cur_nodes]
        env = self.is_env[self.root_idx, self.cur_nodes]

        idx_env = self.root_idx[env & ~leaves_or_terminal & searchable_roots]
        cur_nodes_env = self.cur_nodes[env & ~leaves_or_terminal & searchable_roots]

        refresh = False

        # Expand leaf nodes if enough trees are at a leaf (set to 0 to always expand). 1e-3 is a tolerance for equality
        if (leaves_or_terminal[searchable_roots].float().mean() + 1e-3 >= self.hyperparams.expand_simultaneous_fraction
            # Expand the leaves that currently have the smallest n, that will get us to stop fastest
            # or (leaves_or_terminal.any() and self.n[leaves_or_terminal, 0, 0].amin() == self.n[searchable_roots, 0, 0].amin())
        ):
            # Expand leaves
            self.expand()

            # If we're at a terminal state, reset back to the root
            self.handle_terminal_states()

            refresh = True

        # Advance env nodes, if searchable
        if idx_env.shape[0] > 0:
            self.advance_to_next_states_from_env_states(idx_env, cur_nodes_env)
            refresh = True

        if refresh:
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
        next_actions = self.action_selector.get_actions(self, idx, cur_nodes)

        self.advance_to_next_states(idx, cur_nodes, next_actions)

    def search_for_n_iters(self, n_iters):
        searchable = self.searchable_roots()
        
        # We always at least need to expand the root
        self.expand()

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
        self.next_empty_node_pointers[idx] += 1

    def advance_to_next_states_from_env_states(self, idx, nodes):
        if idx.shape[0] == 0:
            return

        # Now get the next player state for the env states
        legal_action_mask = self.legal_actions_mask[idx, nodes]
        env_action_idx = self.game.get_env_action_idx(self.states[idx, nodes], legal_action_mask)
        next_node_idx = self.next_idx[idx, nodes, env_action_idx]

        # If the next node idx is zero then we need to fill it in
        empty_mask = (next_node_idx == 0)
        empty_idx = idx[empty_mask]
        empty_nodes = nodes[empty_mask]

        next_player_states, is_terminal = self.game.get_next_states_from_env(self.states[empty_idx, empty_nodes], env_action_idx[empty_mask])

        next_nodes = self.next_empty_nodes[empty_idx]
        self.states[empty_idx, next_nodes] = next_player_states
        self.is_terminal[empty_idx, next_nodes] = is_terminal
        self.next_idx[empty_idx, empty_nodes, env_action_idx[empty_mask]] = next_nodes
        self.cur_nodes[empty_idx] = next_nodes
        self.parent_nodes[empty_idx, next_nodes] = empty_nodes
        self.next_empty_node_pointers[empty_idx] += 1

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
        self.player_index[idx, nodes] = self.game.get_cur_player_index(states).to(self.player_index.dtype)
        # Evaluate network and store results
        pi, values = self.evaluator.evaluate(states)
        self.pi[idx, nodes] = pi
        self.values[idx, nodes] = values
        
        player_leaves = is_leaf_mask & ~self.is_env[self.root_idx, self.cur_nodes]
        player_nodes = self.cur_nodes[player_leaves]
        player_idx = self.root_idx[player_leaves]
        player_states = self.states[player_idx, player_nodes]
        
        self.legal_actions_mask[player_idx, player_nodes, :self.n_actions] = self.game.get_legal_action_masks(player_states)
        self.pi[player_idx, player_nodes] *= self.legal_actions_mask[player_idx, player_nodes, :self.n_actions]
        # re-normalize pi
        self.pi[player_idx, player_nodes] /= self.pi[player_idx, player_nodes].sum(dim=1, keepdim=True)
        
        self.logits[player_idx, player_nodes] = torch.log(self.pi[player_idx, player_nodes])

        env_leaves = is_leaf_mask & self.is_env[self.root_idx, self.cur_nodes]
        env_nodes = self.cur_nodes[env_leaves]
        env_idx = self.root_idx[env_leaves]
        env_states = self.states[env_idx, env_nodes]

        if env_nodes.shape[0] > 0:
            self.legal_actions_mask[env_idx, env_nodes, :self.n_stochastic_actions] = self.game.get_env_legal_action_masks(env_states)

        # For any roots, if there is only one valid action, set the flag; prevents searching for efficiency
        self.set_only_one_action(idx, nodes, self.legal_actions_mask[idx, nodes])

        parents = self.parent_nodes[idx, nodes]
        has_parent = parents.to(torch.bool)
        nodesp = nodes[has_parent]
        idxp = idx[has_parent]

        # back up values
        if nodesp.shape[0] > 0:
            self.backup(idxp, self.parent_nodes[idxp, nodesp], self.actions[idxp, nodesp],
                        self.rewards[idxp, nodesp] + (self.hyperparams.discount * values[has_parent]))

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

        self.update_max_min_q(idx, nodes, actions, values)

        parents = self.parent_nodes[idx, nodes]
        actions = self.actions[idx, nodes]
        q = self.rewards[idx, nodes] + self.hyperparams.discount * values
        mask = (nodes != 0)
        self.backup(idx[mask], parents[mask], actions[mask], q[mask])

    def update_max_min_q(self, idx, nodes, actions, values):
        """
        Call this only after updating w and n for the idx, nodes, actions
        """
        # Only possible max Q has gone up if values > maxQ or this is the first value
        # Check those and update if appropriate
        update = (values.amax(dim=1) > self.maxQ[idx]) | torch.isnan(self.maxQ[idx])
        q = self.w[idx[update], nodes[update], actions[update]].amax(dim=1) / self.n[idx[update], nodes[update], actions[update]]
        self.maxQ[idx[update]] = torch.fmax(q, self.maxQ[idx[update]])  # fmax will update nans

        # Only possible min Q has gone down if values < minQ
        update = (values.amin(dim=1) < self.minQ[idx]) | torch.isnan(self.minQ[idx])
        q = self.w[idx[update], nodes[update], actions[update]].amin(dim=1) / self.n[idx[update], nodes[update], actions[update]]
        self.minQ[idx[update]] = torch.fmin(q, self.minQ[idx[update]])

    def add_dirichlet_noise(self):
        self.need_to_add_dirichlet_noise[:] = False
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
        
    def Q(self, idx, cur_nodes, scale_Q: bool = True):
        N = self.n[idx, cur_nodes]
        if scale_Q:
            maxQ = torch.where(torch.isnan(self.maxQ[idx]), self.values[idx, cur_nodes].amax(dim=1), self.maxQ[idx]).unsqueeze(-1)
            minQ = torch.where(torch.isnan(self.minQ[idx]), self.values[idx, cur_nodes].amin(dim=1), self.minQ[idx]).unsqueeze(-1)

            # Q replace unexplored edges with the given state's value
            Q = ((self.w[idx, cur_nodes, :, self.player_index[idx, cur_nodes]]
                  + (N == 0) * self.legal_actions_mask[idx, cur_nodes, :self.n_actions]
                  * self.values[idx, cur_nodes, self.player_index[idx, cur_nodes]].unsqueeze(1))
                 / (N + (N == 0)))

            Q = (Q - minQ + 0.5*(maxQ == minQ))*self.legal_actions_mask[idx, cur_nodes, :self.n_actions] / (maxQ - minQ + (maxQ == minQ))
        else:
            Q = self.w[idx, cur_nodes, :, self.player_index[idx, cur_nodes]] / (N + (N == 0))
            
        return Q

    def get_next_mcts(self, new_roots: torch.Tensor, add_dirichlet_noise=True):
        """
        Preserve as much of the current tree as possible with a new tree whose roots are new_states

        :param new_roots: The new states to start the new tree from
        """
        # First, we need to find the new_roots nodes if they are in the current trees

        state_match = self.states[self.root_idx] == new_roots.unsqueeze(1)
        for _ in range(len(self.state_shape)):
            state_match = state_match.all(dim=-1)

        state_match *= (~self.is_env[self.root_idx])

        # We'll use the first match we find, per root
        new_root_nodes = state_match.int().argmax(dim=1)

        # Move the new roots to position 1, much easier to keep it this way, costs a little time to do these copies
        move = (new_root_nodes != 0)
        self.states[self.root_idx[move], 1] = self.states[self.root_idx[move], new_root_nodes[move]]
        self.legal_actions_mask[self.root_idx[move], 1] = self.legal_actions_mask[self.root_idx[move], new_root_nodes[move]]
        self.pi[self.root_idx[move], 1] = self.pi[self.root_idx[move], new_root_nodes[move]]
        self.values[self.root_idx[move], 1] = self.values[self.root_idx[move], new_root_nodes[move]]
        self.is_terminal[self.root_idx[move], 1] = self.is_terminal[self.root_idx[move], new_root_nodes[move]]
        self.is_leaf[self.root_idx[move], 1] = self.is_leaf[self.root_idx[move], new_root_nodes[move]]
        self.is_env[self.root_idx[move], 1] = self.is_env[self.root_idx[move], new_root_nodes[move]]
        self.next_idx[self.root_idx[move], 1] = self.next_idx[self.root_idx[move], new_root_nodes[move]]
        self.player_index[self.root_idx[move], 1] = self.player_index[self.root_idx[move], new_root_nodes[move]]
        self.w[self.root_idx[move], 1] = self.w[self.root_idx[move], new_root_nodes[move]]
        self.n[self.root_idx[move], 1] = self.n[self.root_idx[move], new_root_nodes[move]]
        new_root_nodes[move] = 1

        # Update parent of children to be 1
        children = self.next_idx[self.root_idx[move], 1]
        self.parent_nodes[self.root_idx[move].unsqueeze(-1).repeat(1, children.shape[1]), children] = 1
        # Reset and parent nodes at 0 to 0
        self.parent_nodes[self.root_idx[move], 0] = 0

        r = self.root_idx.unsqueeze(-1)
        c = new_root_nodes.unsqueeze(-1)
        keep_nodes = [c]

        while True:
            c = self.next_idx[r.repeat(1, c.shape[1]), c].reshape(self.states.shape[0], -1)
            c = c.sort(dim=1, descending=True).values
            non_zero_range = c.argmin(dim=-1).max()

            if non_zero_range == 0:
                break

            c = c[:, :non_zero_range]
            keep_nodes.append(c)

        keep_nodes = torch.cat(keep_nodes, dim=1)
        keep_nodes = keep_nodes.sort(dim=1, descending=True).values

        # Keep nodes now contains all the nodes that will be preserved
        # We need to get the reciprocal nodes, i.e., which in 1:self.max_nodes are not in keep_nodes
        def get_empty_nodes(row_of_keep_nodes):
            empty_nodes_range = torch.arange(1, self.total_states+1)
            match = torch.isin(empty_nodes_range, row_of_keep_nodes, invert=True)
            empty_nodes = empty_nodes_range * match + (self.total_states * ~match)
            # sort
            empty_nodes = empty_nodes.sort().values
            return empty_nodes

        # Use vmap to apply this across the keep_nodes
        self.empty_nodes = torch.vmap(get_empty_nodes)(keep_nodes).to(self.empty_nodes.dtype)
        self.next_empty_node_pointers[:] = 0

        # We need to set all empty nodes as is_leaf
        empty_nodes_capped = torch.minimum(self.empty_nodes, torch.tensor(self.total_states-1, dtype=torch.int))
        roots = r.repeat(1, self.empty_nodes.shape[1])
        self.is_leaf[roots, empty_nodes_capped] = True
        # Reset the next_idx for empty_nodes to 0
        self.next_idx[roots, empty_nodes_capped] = 0
        # Reset w and n
        self.w[roots, empty_nodes_capped] = 0
        self.n[roots, empty_nodes_capped] = 0

        # If new_root_nodes is 0, we don't have the node in the tree, so we need to put it in at empty_nodes
        insert_new_roots = (new_root_nodes == 0)
        assert (self.next_empty_nodes[insert_new_roots] == 1).all()
        self.states[self.root_idx[insert_new_roots], 1] = new_roots[insert_new_roots]
        self.is_leaf[self.root_idx[insert_new_roots], 1] = True  # These states weren't in the tree, need to be expanded
        self.next_empty_node_pointers[insert_new_roots] += 1
        new_root_nodes[insert_new_roots] = 1

        # For the roots we insert, we need to reset w and n to zeros
        self.w[self.root_idx[insert_new_roots], 1] = 0
        self.n[self.root_idx[insert_new_roots], 1] = 0

        assert (new_root_nodes == 1).all().item()

        # Update root nodes
        self.root_nodes = new_root_nodes

        # Clear the parent nodes of the new roots
        self.parent_nodes[self.root_idx, new_root_nodes] = 0

        # Handle terminal roots
        self.handle_terminal_roots()

        # Reset only_one_action
        self.only_one_action[:] = False

        # Start at root
        self.cur_nodes[:] = new_root_nodes

        if add_dirichlet_noise:
            self.add_dirichlet_noise()
            
        self.action_selector.on_mcts_init(self)

        # Update the n row zero
        self.n[:, 0, 0] = self.n[:, 1].sum(dim=1)

        # TODO: update maxQ/minQ ?

        return self
