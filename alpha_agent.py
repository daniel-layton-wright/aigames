from game import *
from tictactoe import *
from connect4 import *
from agent import *
from minimax_agent import *
from manual_agent import *
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AlphaAgent(SequentialAgent):
    def __init__(self, game, evaluator, training_points,
                 tau, c_puct, n_mcts = 1200):
        self.game = game
        self.original_tau = tau
        self.tau = tau
        self.c_puct = c_puct
        self.n_mcts = n_mcts

        self.evaluator = evaluator
        self.cur_node = None
        self.training = True
        self.training_points = training_points

    def choose_action(self, state, player_index, verbose = False):
        if self.cur_node is None:
            self.cur_node = MCTSNode(self.game, state, None, self.evaluator, self.c_puct)
        elif (self.cur_node.state != state).any():
            self.cur_node = MCTSNode(self.game, state, None, self.evaluator, self.c_puct)

        for i in range(self.n_mcts):
            self.cur_node.search()

        pi = np.zeros(len(self.game.ALL_ACTIONS))
        max_N = np.max([child.N for child in self.cur_node.children])
        for action_index, child in zip(self.cur_node.action_indices, self.cur_node.children):
            if self.tau > 0:
                pi[action_index] = child.N**(1./self.tau)
            else:
                pi[action_index] = 1 if child.N == max_N else 0

        pi /= np.sum(pi)
        action_index = np.random.choice(range(len(self.game.ALL_ACTIONS)), p = pi)
        action = self.game.ALL_ACTIONS[action_index]
        child_index = self.cur_node.actions.index(action)

        self.episode_history.append(TimestepData(state, pi, player_index))
        self.cur_node = self.cur_node.children[child_index]
        self.cur_node.parent = None

        return action

    def reward(self, reward_value, state, player_index):
        self.episode_history.append(RewardData(player_index, reward_value))

    def train(self):
        self.tau = self.original_tau
        self.training = True
        self.evaluator.train()

    def eval(self):
        self.tau = 0
        self.training = False
        self.evaluator.eval()

    def start_episode(self):
        self.episode_history = []

    def end_episode(self):
        if not self.training:
            return

        episode_history = reversed(self.episode_history)
        cum_rewards = [0 for _ in range(self.game.N_PLAYERS)]

        for data in episode_history:
            if isinstance(data, RewardData):
                cum_rewards[data.player_index] += data.reward_value
            elif isinstance(data, TimestepData):
                reward = cum_rewards[data.player_index]
                reward = torch.tensor(reward, dtype = torch.float32)
                cur_training_point = (data.state, data.pi, reward)
                self.training_points.append(cur_training_point)


class MCTSNode:
    def __init__(self, game, state, parent_node, evaluator, c_puct):
        self.game = game
        self.state = state
        self.parent = parent_node
        self.actions = None
        self.children = None
        self.evaluator = evaluator
        self.c_puct = c_puct

        self.player_index = self.game.get_player_index(self.state)
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = None
        self.P_normalized = None
        self.v_values = None

    def search(self):
        if self.children is None:
            # We've reached a leaf node, expand
            self.expand()

            # backup
            self.backup(self.v_values)
        else:
            # choose the node according to max Q + U
            U_values = [ self.c_puct * self.P_normalized[i] * np.sqrt(self.N) / (1 + child.N) for i, child in enumerate(self.children) ]
            Q_plus_U_values = [ -node.Q + U_value for node, U_value in zip(self.children, U_values) ]  # -Q because next node is different player
            next_node = self.children[np.argmax(Q_plus_U_values)]

            next_node.search()

    def expand(self):
        if self.game.is_terminal_state(self.state):
            self.v_values = [self.game.reward(self.state, i) for i in range(self.game.N_PLAYERS)]
        else:
            # Evaluate the node with the evaluator
            with torch.no_grad():
                self.P, self.v = self.evaluator(self.state)

            self.v_values = np.repeat(-1 * self.v.detach().numpy(), self.game.N_PLAYERS)
            self.v_values[self.player_index] = self.v

            # Initialize the children
            self.actions = self.game.legal_actions(self.state)
            self.action_indices = sorted([self.game.ALL_ACTIONS.index(action) for action in self.actions])
            self.P_normalized = self.P[self.action_indices].detach().numpy()
            self.P_normalized /= self.P_normalized.sum()

            next_states = [self.game.get_next_state(self.state, action) for action in self.actions]
            self.children = [ MCTSNode(self.game, next_state, self, self.evaluator, self.c_puct) for next_state in next_states ]

    def backup(self, v_values):
        self.N += 1
        self.W += v_values[self.player_index]
        self.Q = self.W / self.N

        if self.parent is not None:
            self.parent.backup(v_values)


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
    def __init__(self, data = None, capacity = None,
                 rotate = False, flip = False):
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


class AlphaTrainingRun:
    def __init__(self, game_class, evaluator,
                 tau = 1, c_puct = 1, n_mcts = 500,
                 lr = 1e-2, wd = 1e-4,
                 rotate = False, flip = False):
        self.game_class = game_class
        self.evaluator = evaluator

        self.training_data = []
        self.training_dataset = AlphaDataset(self.training_data, capacity = 50*1000, rotate=rotate, flip = flip)

        self.tau = tau
        self.agent = AlphaAgent(self.game_class, self.evaluator, self.training_dataset, tau, c_puct, n_mcts)
        self.minimax_agent = MinimaxAgent(self.game_class)
        self.optimizer = torch.optim.Adam(self.evaluator.parameters(), lr = lr, weight_decay=wd)

    def self_play_game(self):
        game = self.game_class([self.agent, self.agent])
        game.play()
        return game

    def take_training_step(self, batch_size = 32):
        # Clear the gradients
        self.evaluator.zero_grad()

        # Sample batch from training points
        dataloader = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True)
        states, distns, values = next(iter(dataloader))

        # Run through network
        pred_distns, pred_values = self.evaluator(states)

        # Compute loss
        losses = (values - pred_values)**2 - torch.sum(distns * torch.log(pred_distns), dim = 1)
        loss = torch.mean(losses)

        # Backprop
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, n_games = 100):
        n_losses = 0

        self.agent.eval()

        for _ in range(n_games):
            game = self.game_class([self.minimax_agent, self.agent])
            game.play()
            if game.reward(game.state, 1) == -1:
                n_losses += 1

        self.agent.train()

        return n_losses / float(n_games)

    def run(self, n_games = 20000, debugger = None):
        for n_game in range(n_games):
            cur_game = self.self_play_game()
            cur_loss = self.take_training_step()

            if debugger is not None:
                debugger(self, cur_game, cur_loss)
