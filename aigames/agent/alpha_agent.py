import numpy as np
from .agent import Agent
from ..game.game import SequentialGame
from typing import Type, List
from tqdm.auto import tqdm


class BaseAlphaEvaluator:
    def evaluate(self, state):
        raise NotImplementedError()

    def process_state(self, state):
        raise NotImplementedError()

    def eval(self):
        # Switch to eval mode
        pass

    def train(self):
        # Switch to train mode
        pass


class AlphaAgentListener:
    def after_mcts_search(self, root_node):
        pass

    def on_data_point(self, state, pi, reward):
        pass


class AlphaAgentDebugListener(AlphaAgentListener):
    def after_mcts_search(self, root_node):
        import pdb
        pdb.set_trace()


class TrainingTau:
    def __init__(self, fixed_tau_value=None, tau_schedule_list=None, tau_schedule_function=None):
        if sum(map(int, [fixed_tau_value is not None, tau_schedule_list is not None, tau_schedule_function is not None])) != 1:
            raise ValueError('Please pass exactly one of fixed_tau_value, tau_schedule_list, tau_schedule_function')

        self.fixed_tau_value = fixed_tau_value
        self.tau_schedule_list = tau_schedule_list
        self.tau_schedule_function = tau_schedule_function

    def get_tau(self, move_number):
        if self.fixed_tau_value is not None:
            return self.fixed_tau_value
        elif self.tau_schedule_list is not None:
            return self.tau_schedule_list[min(len(self.tau_schedule_list) - 1, move_number)]
        else:
            return self.tau_schedule_function(move_number)


class AlphaAgentHyperparameters:
    __slots__ = ['c_puct', 'dirichlet_alpha', 'dirichlet_epsilon', 'n_mcts', 'discount_rate', 'training_tau']

    def __init__(self):
        self.c_puct = 1.0
        self.dirichlet_alpha = 0.3
        self.dirichlet_epsilon = 0.25
        self.n_mcts = 1200
        self.discount_rate = 0.99
        self.training_tau = TrainingTau(1.0)


class AlphaAgent(Agent):
    def __init__(self, game_class: Type[SequentialGame], evaluator: BaseAlphaEvaluator, hyperparams: AlphaAgentHyperparameters,
                 listeners: List[AlphaAgentListener] = None, use_tqdm: bool = False):
        super().__init__()
        self.game_class = game_class
        self.all_actions = self.game_class.get_all_actions()
        self.hyperparams = hyperparams
        self.evaluator = evaluator
        self.cur_node = None
        self.training = True
        self.episode_history = []
        self.move_number_in_current_game = 0
        self.n_players = 0
        self.listeners = listeners if listeners is not None else []
        self.use_tqdm = use_tqdm

    def get_action(self, state, legal_actions):
        # Try to re-use the current node if possible but if it is None or does not match the current state, initialize
        # a new one from scratch
        if self.cur_node is None or self.cur_node.state is None or (not self.game_class.states_equal(self.cur_node.state, state)):
            self.cur_node = MCTSNode(self.game_class, state, None, self.evaluator, self.hyperparams, n_players=self.n_players)

        # Add Dirichlet noise to the root node
        self.cur_node.add_dirichlet_noise()

        r = range(self.hyperparams.n_mcts)
        if self.use_tqdm:
            r = tqdm(r)

        # Do MCTS search
        for i in r:
            self.cur_node.search()

        for listener in self.listeners:
            listener.after_mcts_search(self.cur_node)

        if self.training:
            tau = self.hyperparams.training_tau.get_tau(self.move_number_in_current_game)
        else:
            tau = 0.0

        try:
            # Compute the distribution to choose from
            pi = np.zeros(len(self.all_actions))

            if self.hyperparams.n_mcts > 1:
                action_distribution = self.cur_node.children_total_N
            else:
                action_distribution = self.cur_node.P_normalized

            if tau > 0:
                pi[self.cur_node.action_indices] = action_distribution ** (1. / tau)
                pi /= np.sum(pi)
            else:
                pi[self.cur_node.action_indices[np.argmax(self.cur_node.children_total_N)]] = 1

            # Choose the action according to the distribution pi
            action_index = np.random.choice(range(len(self.all_actions)), p=pi)
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()

        action = self.all_actions[action_index]
        legal_action_index = legal_actions.index(action)
        child_index = self.cur_node.actions.index(action)

        # Record this for training, set the root node to the next node now and forget the parent
        player_index = self.game_class.get_cur_player_index(state)
        self.episode_history.append(TimestepData(state, pi, player_index))

        self.cur_node = self.cur_node.children[child_index]
        self.cur_node.parent = None
        self.move_number_in_current_game += 1

        return legal_action_index

    def on_reward(self, reward, next_state, player_index):
        self.episode_history.append(RewardData(player_index, reward))

        try:
            if self.cur_node is not None and (not self.game_class.states_equal(self.cur_node.state, next_state)):
                next_state_index = next(i for i, x in enumerate(self.cur_node.children)
                                        if self.game_class.states_equal(x.state, next_state))
                self.cur_node = self.cur_node.children[next_state_index]
                self.cur_node.parent = None
        except Exception as e:
            print("MCTSNode.on_reward : next_state does not exist in children states")

    def train(self):
        self.evaluator.train()
        self.training = True

    def eval(self):
        self.evaluator.eval()
        self.training = False

    def before_game_start(self, n_players):
        self.n_players = n_players
        self.episode_history = []
        self.move_number_in_current_game = 0
        self.cur_node = None

    def on_game_end(self):
        if not self.training:
            return

        episode_history = reversed(self.episode_history)  # work backwards
        cum_rewards = [0 for _ in range(self.n_players)]

        for data in episode_history:
            if isinstance(data, RewardData):
                cum_rewards[data.player_index] = data.reward_value + self.hyperparams.discount_rate * cum_rewards[data.player_index]
            elif isinstance(data, TimestepData):
                reward = cum_rewards[data.player_index]
                for data_listener in self.listeners:
                    data_listener.on_data_point(data.state, data.pi, reward)


class MCTSNode:
    def __init__(self, game_class: Type[SequentialGame], state, parent_node, evaluator: BaseAlphaEvaluator,
                 hyperparams: AlphaAgentHyperparameters, n_players=2, action_index_from_parent_node=None,
                 N=None, total_N=None, Q=None):
        self.game_class = game_class
        self.state = state
        self.parent = parent_node
        self.action_index_from_parent_node = action_index_from_parent_node
        self.all_actions = self.game_class.get_all_actions()
        self.actions = None
        self.action_indices = None
        self.n_children = None
        self.n_players = n_players
        self.children = None
        self.children_N = None
        self.children_total_N = None
        self.children_Q = None
        self.evaluator = evaluator
        self.hyperparams = hyperparams
        self.player_index = None

        if self.state is None:  # this is a node created from a parent, let's get the state
            self.state, self.rewards = self.game_class.get_next_state_and_rewards(self.parent.state,
                                                                                  self.parent.actions[self.action_index_from_parent_node])

        self.is_terminal_state = self.game_class.is_terminal_state(self.state)

        self.N = N if N is not None else np.zeros(self.n_players)
        self.total_N = total_N if total_N is not None else np.array([0])
        self.Q = Q if Q is not None else np.zeros(self.n_players)

        if self.action_index_from_parent_node is None:
            self.rewards = self.game_class.get_rewards(self.state)  # this is a brand new node, let's get the rewards

        self.W = np.zeros(self.n_players)
        self.P = None
        self.P_normalized = None
        self._need_to_add_dirichlet_noise = False
        self.v = None  # v is from the perspective of the player whose turn it is

    def fill_info(self):
        # call this when self.state is not None
        self.actions = self.game_class.get_legal_actions(self.state)
        self.action_indices = [self.all_actions.index(action) for action in self.actions]
        self.n_children = len(self.actions)
        self.children_N = np.zeros((self.n_children, self.n_players))
        self.children_total_N = np.zeros(self.n_children)
        self.children_Q = np.zeros((self.n_children, self.n_players))
        self.player_index = self.game_class.get_cur_player_index(self.state)

    def search(self):
        if self.actions is None:  # we haven't yet filled in all this info, let's do it now
            self.fill_info()

        if self.children is None:
            # We've reached a leaf node, expand
            self.expand()

            # backup
            if not self.is_terminal_state:
                self.backup(self.v, self.player_index)
            else:
                self.backup_all(self.rewards)
        else:
            # choose the node according to max Q + U
            U_values = self.hyperparams.c_puct * self.P_normalized * np.sqrt(self.total_N) / (1 + self.children_total_N)
            Q_plus_U_values = self.children_Q[:, self.player_index] + U_values
            next_node = self.children[np.argmax(Q_plus_U_values)]
            next_node.search()

    def expand(self):
        if not self.is_terminal_state:
            # Evaluate the node with the evaluator
            self.P, self.v = self.evaluator.evaluate(self.state)

            # Filter the distribution to allowed actions
            self.P_normalized = self.P[self.action_indices] #.detach().cpu().numpy()
            self.P_normalized /= self.P_normalized.sum()
            if self._need_to_add_dirichlet_noise:
                self._add_dirichlet_noise()

            # Initialize the children
            self.children = [
                MCTSNode(self.game_class, None, self, self.evaluator, self.hyperparams, self.n_players,
                         action_index_from_parent_node=i,
                         N=self.children_N[i], total_N=self.children_total_N[i:(i + 1)],
                         Q=self.children_Q[i])
                for i in range(len(self.actions))
                ]

    def add_dirichlet_noise(self):
        if self.P_normalized is not None:
            self._add_dirichlet_noise()
        else:
            self._need_to_add_dirichlet_noise = True

    def _add_dirichlet_noise(self):
        self.P_normalized = ((1 - self.hyperparams.dirichlet_epsilon) * self.P_normalized +
                             self.hyperparams.dirichlet_epsilon * np.random.dirichlet([self.hyperparams.dirichlet_alpha] * self.n_children))

    def backup(self, value, player_index):
        self.N[player_index] += 1
        self.total_N += 1
        self.W[player_index] += value
        self.Q[player_index] = self.rewards[player_index] + self.W[player_index] / self.N[player_index]

        if self.parent is not None:
            self.parent.backup(value + self.rewards[player_index], player_index)

    def backup_all(self, values):
        self.N += 1
        self.total_N += 1
        self.W += values
        self.Q[:] = self.rewards + self.W / self.N
        if self.parent is not None:
            self.parent.backup_all(values)


class TimestepData:
    def __init__(self, state, pi, player_index):
        self.state = state
        self.pi = pi
        self.player_index = player_index

    def __repr__(self):
        return f'(state: {self.state}, pi={self.pi}, i={self.player_index})'


class RewardData:
    def __init__(self, player_index, reward_value):
        self.player_index = player_index
        self.reward_value = reward_value

    def __repr__(self):
        return f'(p: {self.player_index}, r: {self.reward_value})'


class DummyAlphaEvaluator(BaseAlphaEvaluator):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.P = np.ones(n_actions)
        self.P /= self.P.sum()

    def evaluate(self, state):
        return self.P, 0

    def process_state(self, state):
        return state

    def eval(self):
        # Switch to eval mode
        pass

    def train(self):
        # Switch to train mode
        pass
