import numpy as np
import torch
from aigames.base.game import *
from aigames.base.agent import *
from aigames.alpha_evaluator import AlphaEvaluator
from torch.utils.data import Dataset


# TODO : add Dirichlet noise
class AlphaAgent(SequentialAgent):
    def __init__(self, game: Game, evaluator: AlphaEvaluator,
                 training_tau: float, c_puct: float, n_mcts=1200,
                 discount_rate: float=1):
        super().__init__(game)
        self.game = game
        self.training_tau = training_tau
        self.tau = training_tau
        self.c_puct = c_puct
        self.n_mcts = n_mcts
        self.discount_rate = discount_rate

        self.evaluator = evaluator
        self.cur_node = None
        self.training = True

    def choose_action(self, state, player_index, verbose = False):
        if self.cur_node is None:
            self.cur_node = MCTSNode(self.game, state, None, self.evaluator, self.c_puct)
        elif (self.cur_node.state != state).any():
            self.cur_node = MCTSNode(self.game, state, None, self.evaluator, self.c_puct)

        for i in range(self.n_mcts):
            self.cur_node.search()

        pi = np.zeros(len(self.game.ALL_ACTIONS))
        if self.tau > 0:
            pi[self.cur_node.action_indices] = self.cur_node.children_total_N**(1./self.tau)
            pi /= np.sum(pi)
        else:
            pi[self.cur_node.action_indices[np.argmax(self.cur_node.children_total_N)]] = 1

        action_index = np.random.choice(range(len(self.game.ALL_ACTIONS)), p=pi)
        action = self.game.ALL_ACTIONS[action_index]
        child_index = self.cur_node.actions.index(action)

        self.episode_history.append(TimestepData(state, pi, player_index))
        self.cur_node = self.cur_node.children[child_index]
        self.cur_node.parent = None

        return action

    def reward(self, reward_value, state, player_index):
        self.episode_history.append(RewardData(player_index, reward_value))

    def train(self):
        self.tau = self.training_tau
        self.training = True

    def eval(self):
        self.tau = 0
        self.training = False

    def start_episode(self):
        self.episode_history = []

    def end_episode(self):
        if not self.training:
            return

        episode_history = reversed(self.episode_history)
        cum_rewards = [0 for _ in range(self.game.N_PLAYERS)]

        states = []
        pis = []
        rewards = []

        for data in episode_history:
            if isinstance(data, RewardData):
                cum_rewards[data.player_index] = data.reward_value + self.discount_rate * cum_rewards[data.player_index]
            elif isinstance(data, TimestepData):
                reward = cum_rewards[data.player_index]

                states.append(data.state)
                pis.append(data.pi)
                rewards.append(torch.FloatTensor([reward]))

        self.evaluator.train(states, pis, rewards)


class MCTSNode:
    def __init__(self, game, state, parent_node, evaluator, c_puct, N=None, total_N=None, Q=None):
        self.game = game
        self.state = state
        self.parent = parent_node
        self.actions = self.game.legal_actions(self.state)
        self.next_states = [self.game.get_next_state(self.state, action) for action in self.actions]
        self.action_indices = self.game.legal_action_indices(self.state)
        self.n_children = len(self.actions)
        self.children = None
        self.children_N = np.zeros((self.n_children, self.game.N_PLAYERS))
        self.children_total_N = np.zeros(self.n_children)
        self.children_Q = np.zeros((self.n_children, self.game.N_PLAYERS))
        self.evaluator = evaluator
        self.c_puct = c_puct

        self.player_index = self.game.get_player_index(self.state)
        self.N = N if N is not None else np.zeros(self.game.N_PLAYERS)
        self.total_N = total_N if total_N is not None else np.array([0])
        self.Q = Q if Q is not None else np.zeros(self.game.N_PLAYERS)

        self.rewards = self.game.all_rewards(self.state)
        self.W = np.zeros(self.game.N_PLAYERS)
        self.P = None
        self.P_normalized = None
        self.v = None  # v is from the perspective of the player whose turn it is

    def search(self):
        if self.children is None:
            # We've reached a leaf node, expand
            self.expand()

            # backup
            if not self.game.is_terminal_state(self.state):
                self.backup(self.v, self.player_index)
            else:
                self.backup_all(self.rewards)
        else:
            # choose the node according to max Q + U
            U_values = self.c_puct * self.P_normalized * np.sqrt(self.total_N) / (1 + self.children_total_N)
            Q_plus_U_values = self.children_Q[:,self.player_index] + U_values
            next_node = self.children[np.argmax(Q_plus_U_values)]
            next_node.search()

    def expand(self):
        if not self.game.is_terminal_state(self.state):
            # Evaluate the node with the evaluator
            self.P, self.v = self.evaluator.evaluate(self.state)

            # Initialize the children
            self.P_normalized = self.P[self.action_indices].detach().numpy()
            self.P_normalized /= self.P_normalized.sum()

            self.children = [ MCTSNode(self.game, next_state, self, self.evaluator, self.c_puct,
                                       self.children_N[i], self.children_total_N[i:(i+1)], self.children_Q[i])
                              for i, next_state in enumerate(self.next_states) ]

    def backup(self, value, player_index):
        self.N[player_index] += 1
        self.total_N += 1
        self.W[player_index] += value
        self.Q[player_index] = self.rewards[player_index] + self.W[player_index] / self.N[player_index]

        if self.parent is not None:
            self.parent.backup(value, player_index + self.rewards[player_index])

    def backup_all(self, values):
        self.N += 1
        self.total_N += 1
        self.W += values
        self.Q[:] = self.rewards + self.W / self.N
        if self.parent is not None:
            self.parent.backup_all(values)


class TimestepData:
    def __init__(self, state, pi, player_index):
        self.state = torch.FloatTensor(state)
        self.pi = torch.FloatTensor(pi)
        self.player_index = player_index


class RewardData:
    def __init__(self, player_index, reward_value):
        self.player_index = player_index
        self.reward_value = reward_value


class AlphaDataset(Dataset):
    """
    The dataset is special in that it is built up bit by bit as the self-play games are played
    """
    def __init__(self, data=None, capacity=None, rotate=False, flip=False):
        super().__init__()
        self.data = data if data is not None else data

        self.rotate = rotate
        self.flip = flip
        self.capacity = capacity

    def append(self, tuple):
        """

        :param tuple: (state, distn, value)
        :return: None
        """
        self.data.append(tuple)

        if self.capacity is not None and len(self.data) > self.capacity:
            self.data.pop(0)

    def __getitem__(self, i):
        return self.preprocess(*self.data[i])

    def __len__(self):
        return len(self.data)

    def preprocess(self, state, distn, value):
        r = np.random.randint(8)
        n_rots = r % 4
        flip =  bool(r / 4)

        new_state = state
        new_distn = distn

        if self.rotate:
            new_state = new_state.rot90(k = n_rots, dims = (1,2))
            new_distn = new_distn.reshape(-1, *state.shape[1:]).rot90(k = n_rots, dims = (1,2)).reshape(distn.shape)

        if self.flip and flip:
            new_state = new_state.flip(dims = (2,))
            new_distn = new_distn.reshape(-1, *state.shape[1:]).flip(dims = (2,)).reshape(distn.shape)

        return new_state, new_distn, value
