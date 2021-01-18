import numpy as np
from .agent import Agent
from ..game.game import SequentialGame, PartiallyObservableSequentialGame
from typing import Type
import torch


class BaseAlphaEvaluator:
    def evaluate(self, state):
        raise NotImplementedError()

    def take_training_step(self, states, action_distns, values):
        raise NotImplementedError()

    def eval(self):
        # Switch to eval mode
        pass

    def train(self):
        # Switch to train mode
        pass


class AlphaAgentDataListener:
    def on_data_point(self, state, pi, reward):
        raise NotImplementedError()


class TrainingTau:
    def __init__(self, fixed_tau_value=None, tau_schedule_list=None, tau_schedule_function=None):
        if sum(map(int, fixed_tau_value is not None, tau_schedule_list is not None, tau_schedule_function is not None)) != 1:
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


class AlphaAgent(Agent):
    def __init__(self, game_class: Type[PartiallyObservableSequentialGame], evaluator: BaseAlphaEvaluator,
                 data_listener: AlphaAgentDataListener,
                 training_tau: TrainingTau, c_puct: float = 1.,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25, n_mcts: int = 1200,
                 discount_rate: float = 1):
        super().__init__()
        self.game_class = game_class
        self.all_actions = self.game_class.get_all_actions()
        self.training_tau = training_tau
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.n_mcts = n_mcts
        self.discount_rate = discount_rate
        self.evaluator = evaluator
        self.cur_node = None
        self.training = True
        self.episode_history = []
        self.move_number_in_current_game = 0
        self.n_players = 0
        self.data_listener = data_listener

    def get_action(self, state, legal_actions):
        # Try to re-use the current node if possible but if it is None or does not match the current state, initialize
        # a new one from scratch
        if self.cur_node is None or (self.cur_node.state != state).any():
            if self.cur_node is not None:
                print('Having to re-init node...', state)
            self.cur_node = MCTSNode(self.game_class, state, None, self.evaluator, self.c_puct, self.dirichlet_alpha,
                                     self.dirichlet_epsilon, n_players=self.n_players)

        # Add Dirichlet noise to the root node
        self.cur_node.add_dirichlet_noise()

        # Do MCTS search
        for i in range(self.n_mcts):
            self.cur_node.search()

        if self.training:
            self.tau = self.training_tau.get_tau(self.move_number_in_current_game)

        # Compute the distribution to choose from
        pi = np.zeros(len(self.all_actions))
        if self.tau > 0:
            pi[self.cur_node.action_indices] = self.cur_node.children_total_N ** (1. / self.tau)
            pi /= np.sum(pi)
        else:
            pi[self.cur_node.action_indices[np.argmax(self.cur_node.children_total_N)]] = 1

        # Choose the action according to the distribution pi
        action_index = np.random.choice(range(len(self.all_actions)), p=pi)
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
        if self.cur_node is not None and ((self.cur_node.state != next_state).any()):
            try:
                next_state_index = next(i for i, x in enumerate(self.cur_node.next_states) if np.array_equal(next_state, x))
            except StopIteration:
                print('State not found', next_state)
                pass
            else:
                self.cur_node = self.cur_node.children[next_state_index]
                self.cur_node.parent = None

    def train(self):
        self.tau = self.training_tau_schedule[0]
        self.evaluator.train()
        self.training = True

    def eval(self):
        self.tau = 0
        self.evaluator.eval()
        self.training = False

    def before_game_start(self, n_players):
        self.n_players = n_players
        self.episode_history = []
        self.move_number_in_current_game = 0
        self.cur_node = None

    def end_episode(self):
        if not self.training:
            return

        episode_history = reversed(self.episode_history)  # work backwards
        cum_rewards = [0 for _ in range(self.n_players)]
        states = []
        pis = []
        rewards = []

        for data in episode_history:
            if isinstance(data, RewardData):
                cum_rewards[data.player_index] = data.reward_value + self.discount_rate * cum_rewards[data.player_index]
            elif isinstance(data, TimestepData):
                reward = cum_rewards[data.player_index]
                self.data_listener.on_data_point(data.state, data.pi, reward)


class MCTSNode:
    def __init__(self, game_class, state, parent_node, evaluator: BaseAlphaEvaluator, c_puct, dirichlet_alpha,
                 dirichlet_noise_weight: float = 0., n_players=2,
                 N=None, total_N=None, Q=None):
        self.game_class = game_class
        self.state = state
        self.parent = parent_node
        self.all_actions = self.game_class.get_all_actions()
        self.actions = self.game_class.get_legal_actions(self.state)
        self.next_states_and_rewards = [self.game_class.get_next_state_and_rewards(self.state, action) for action in self.actions]
        self.next_states = [state for state, action in self.next_states_and_rewards]
        self.action_indices = [self.all_actions.index(action) for action in self.actions]
        self.n_children = len(self.actions)
        self.n_players = n_players
        self.children = None
        self.children_N = np.zeros((self.n_children, self.n_players))
        self.children_total_N = np.zeros(self.n_children)
        self.children_Q = np.zeros((self.n_children, self.n_players))
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_noise_weight = dirichlet_noise_weight

        self.player_index = self.game_class.get_cur_player_index(self.state)
        self.N = N if N is not None else np.zeros(self.n_players)
        self.total_N = total_N if total_N is not None else np.array([0])
        self.Q = Q if Q is not None else np.zeros(self.n_players)

        self.rewards = self.game_class.get_rewards(self.state)
        self.W = np.zeros(self.n_players)
        self.P = None
        self.P_normalized = None
        self._need_to_add_dirichlet_noise = False
        self.v = None  # v is from the perspective of the player whose turn it is

    def search(self):
        if self.children is None:
            # We've reached a leaf node, expand
            self.expand()

            # backup
            if not self.game_class.is_terminal_state(self.state):
                self.backup(self.v, self.player_index)
            else:
                self.backup_all(self.rewards)
        else:
            # choose the node according to max Q + U
            U_values = self.c_puct * self.P_normalized * np.sqrt(self.total_N) / (1 + self.children_total_N)
            Q_plus_U_values = self.children_Q[:, self.player_index] + U_values
            next_node = self.children[np.argmax(Q_plus_U_values)]
            next_node.search()

    def expand(self):
        if not self.game_class.is_terminal_state(self.state):
            # Evaluate the node with the evaluator
            self.P, self.v = self.evaluator.evaluate(self.state)

            # Filter the distribution to allowed actions
            self.P_normalized = self.P[self.action_indices] #.detach().cpu().numpy()
            self.P_normalized /= self.P_normalized.sum()
            if self._need_to_add_dirichlet_noise:
                self._add_dirichlet_noise()

            # Initialize the children
            self.children = [
                MCTSNode(self.game_class, next_state, self, self.evaluator, self.c_puct, self.dirichlet_alpha,
                         self.dirichlet_noise_weight, self.n_players, self.children_N[i], self.children_total_N[i:(i + 1)],
                         self.children_Q[i])
                for i, next_state in enumerate(self.next_states)
                ]

    def add_dirichlet_noise(self):
        if self.P_normalized is not None:
            self._add_dirichlet_noise()
        else:
            self._need_to_add_dirichlet_noise = True

    def _add_dirichlet_noise(self):
        self.P_normalized = ((1 - self.dirichlet_noise_weight) * self.P_normalized +
                             self.dirichlet_noise_weight * np.random.dirichlet([self.dirichlet_alpha] * self.n_children))

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


class RewardData:
    def __init__(self, player_index, reward_value):
        self.player_index = player_index
        self.reward_value = reward_value


class DummyAlphaEvaluator(BaseAlphaEvaluator):
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.P = np.ones(n_actions)
        self.P /= self.P.sum()

    def evaluate(self, state):
        return self.P, 0

    def take_training_step(self, states, action_distns, values):
        pass

    def eval(self):
        # Switch to eval mode
        pass

    def train(self):
        # Switch to train mode
        pass
