import numpy as np
import copy
import torch
import torch.nn as nn
from collections import defaultdict
from agent import *


class QLearningAgent(SequentialAgent):
    def __init__(self, game, Q, exploration_probability = 0.1,
                 batch_size = 16, lr = 0.01, update_target_Q_every = 10000,
                 min_replay_memory_size = 10000, max_replay_memory_size = 10000):
        """

        :param Q: the Q predictor to be used
        """
        self.game = game
        self.exploration_probability = exploration_probability
        self.training = True

        self.Q = Q
        self.target_Q = copy.deepcopy(Q)
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr = lr)

        self.replay_memory = []
        self.min_replay_memory_size = min_replay_memory_size
        self.max_replay_memory_size = max_replay_memory_size
        assert(self.max_replay_memory_size >= self.min_replay_memory_size)

        self.loss_ema_param = 0.99
        self.loss_ema = None
        self.loss_history = []
        self.loss_ema_history = []
        self.n_iters = 0
        self.update_target_Q_every = update_target_Q_every

        # Used to save so that when the reward comes back we can add to the replay memory
        self.prev_state = {}
        self.prev_processed_state = {}
        self.prev_action_index = {}
        self.prev_rewards = defaultdict(int)

    def choose_action(self, state, player_index, verbose = False):
        legal_actions = self.game.legal_actions(state)
        legal_action_indices = [self.game.ALL_ACTIONS.index(a) for a in legal_actions]

        # epsilon-Greedy
        if (self.training and np.random.random() < self.exploration_probability) or len(self.replay_memory) < self.min_replay_memory_size:
            # Explore
            idx = np.random.choice(legal_action_indices)
        else:
            # Exploit
            with torch.no_grad():
                scores = []
                for action in legal_actions:
                    processed_state_action = self.Q.process_state_action(state, action, player_index)
                    scores.append(self.Q(processed_state_action))

                idx = legal_action_indices[np.argmax(scores)]

        self.prev_state[player_index] = copy.deepcopy(state)
        self.prev_action_index[player_index] = idx

        return self.game.ALL_ACTIONS[idx]

    def max_over_actions_target_Q(self, state, player_index):
        scores = []
        legal_actions = self.game.legal_actions(state)
        for action in legal_actions:
            processed_state_action = self.Q.process_state_action(state, action, player_index)
            with torch.no_grad():
                scores.append(self.target_Q(processed_state_action))

        return max(scores).flatten()

    def reward(self, reward_value, next_state, player_index):
        # Don't do anything if this is just the initial reward
        if player_index not in self.prev_state or not self.training:
            return

        if player_index != self.game.get_player_index(next_state) and not self.game.is_terminal_state(next_state):
            # just add this reward to be included when we get back to this player
            self.prev_rewards[player_index] += reward_value
            return

        # Add to replay memory if we're back to the given player's turn
        self.replay_memory.append(
            {
                'prev_state': self.prev_state[player_index],
                'prev_action_index': self.prev_action_index[player_index],
                'next_state': copy.deepcopy(next_state),
                'reward': self.prev_rewards[player_index] + reward_value,
                'next_state_is_terminal': self.game.is_terminal_state(next_state)
            }
        )
        # Reset running rewards to 0
        self.prev_rewards[player_index] = 0

        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        while len(self.replay_memory) > self.max_replay_memory_size:
            self.replay_memory.pop(0)

        if (self.n_iters + 1) % self.update_target_Q_every == 0:
            self.target_Q = copy.deepcopy(self.Q)
            self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr = self.lr)

        # Sample a minibatch and train
        train_minibatch = np.random.choice(self.replay_memory, self.batch_size)
        train_minibatch_terminal = [x for x in train_minibatch if x['next_state_is_terminal']]
        train_minibatch_non_terminal = [x for x in train_minibatch if not x['next_state_is_terminal']]
        train_minibatch = train_minibatch_terminal + train_minibatch_non_terminal

        batch_prev_processed_states = torch.cat(
            tuple(
                self.Q.process_state_action(x['prev_state'], self.game.ALL_ACTIONS[x['prev_action_index']], player_index) for x in train_minibatch
            )
        )
        prev_action_indices = [x['prev_action_index'] for x in train_minibatch]
        rewards = (torch.tensor([x['reward'] for x in train_minibatch]).to(torch.float) )

        best_vals = []
        for non_terminal in train_minibatch_non_terminal:
            best_vals.append(self.max_over_actions_target_Q(non_terminal['next_state'], player_index))

        if len(train_minibatch_non_terminal) > 0:
            best_vals = torch.cat(best_vals)
            next_state_q = torch.cat((torch.tensor(np.zeros((len(train_minibatch_terminal),))).to(torch.float), best_vals)).flatten()
        else:
            next_state_q = torch.tensor(np.zeros((len(train_minibatch_terminal),))).to(torch.float).flatten()

        y_target = rewards + next_state_q

        self.Q.zero_grad()
        predicted_values = self.Q(batch_prev_processed_states)
        loss = ((y_target - predicted_values)**2).mean()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())
        if self.loss_ema is None:
            self.loss_ema = loss.item()
        else:
            self.loss_ema = self.loss_ema_param * self.loss_ema + (1 - self.loss_ema_param) * loss.item()
        self.loss_ema_history.append(self.loss_ema)

        self.n_iters += 1


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Qconv(nn.Module):
    def __init__(self, game, n_filters = (64, 16)):
        super().__init__()
        self.game = game
        self.network = nn.Sequential(
            torch.nn.Conv2d(1, n_filters[0], 3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.Conv2d(n_filters[0], n_filters[1], 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=(n_filters[1] * 3 * 3), out_features=1, bias=True)
        )

    def forward(self, processed_state):
        return self.network(processed_state)

    def process_state_action(self, state, action, player_index):
        x = (2*player_index - 1) * self.game.get_next_state(state, action)
        x = torch.tensor(x).to(torch.float)
        x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        return x

class Q(nn.Module):
    def __init__(self, game, hidden_size, activation_fn):
        super().__init__()
        self.game = game
        if not hasattr(hidden_size, '__len__'):
            hidden_size = (hidden_size,)
        self.hidden_size = hidden_size
        self.n_layers = len(hidden_size)

        layers = [nn.Linear(game.STATE_SIZE + game.ACTION_SIZE + 1, self.hidden_size[0])]
        for i in range(1, self.n_layers):
            layers.append(activation_fn())
            layers.append(nn.Linear(self.hidden_size[i-1], self.hidden_size[i]))

        layers.append(activation_fn())
        layers.append(nn.Linear(self.hidden_size[-1], 1))
                                #len(self.game.ALL_ACTIONS)))

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, processed_state):
        pred_scores = self.network(processed_state)
        return pred_scores

    @staticmethod
    def process_state_action(state, action, player_index):
        out = torch.tensor(np.concatenate((state.flatten(), np.array(action).flatten(), np.array([player_index])))).to(torch.float)
        return out
