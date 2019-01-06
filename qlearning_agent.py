import numpy as np
import copy
import torch
import torch.nn as nn
from collections import defaultdict
from agent import *


class QLearningAgent(SequentialAgent):
    def __init__(self, game, Q, exploration_probability = 0.1,
                 batch_size = 16, lr = 0.01, update_target_Q_every = 1000):
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
        processed_state = self.Q.process_state(state, player_index)

        # epsilon-Greedy
        if self.training and np.random.random() < self.exploration_probability:
            # Explore
            idx = np.random.choice(len(self.game.ALL_ACTIONS))
        else:
            # Exploit
            with torch.no_grad():
                pred_scores = self.Q(processed_state)

                if verbose:
                    print('Have I seen this before? {}'.format(
                        any((x['prev_state'] == state).all() for x in self.replay_memory)
                    ))
                    print(dict((action, score.item()) for action, score in zip(self.game.ALL_ACTIONS, pred_scores.data)))

                # Choose the best LEGAL index
                legal_actions = self.game.legal_actions(state)
                legal_action_indices = [self.game.ALL_ACTIONS.index(a) for a in legal_actions]
                legal_scores = pred_scores[legal_action_indices]
                _, best_legal_action_idx = legal_scores.topk(1)
                idx = legal_action_indices[best_legal_action_idx]

        self.prev_state[player_index] = copy.deepcopy(state)
        self.prev_processed_state[player_index] = processed_state
        self.prev_action_index[player_index] = idx

        return self.game.ALL_ACTIONS[idx]

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
                'prev_processed_state': self.prev_processed_state[player_index],
                'prev_action_index': self.prev_action_index[player_index],
                'next_state': copy.deepcopy(next_state),
                'next_processed_state': self.Q.process_state(next_state, player_index),
                'reward': self.prev_rewards[player_index] + reward_value,
                'next_state_is_terminal': self.game.is_terminal_state(next_state)
            }
        )
        # Reset running rewards to 0
        self.prev_rewards[player_index] = 0

        if len(self.replay_memory) == 0:
            return

        if (self.n_iters + 1) % self.update_target_Q_every == 0:
            self.target_Q = copy.deepcopy(self.Q)
            self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr = self.lr)

        # Sample a minibatch and train
        train_minibatch = np.random.choice(self.replay_memory, self.batch_size)
        train_minibatch_terminal = [x for x in train_minibatch if x['next_state_is_terminal']]
        train_minibatch_non_terminal = [x for x in train_minibatch if not x['next_state_is_terminal']]
        train_minibatch = train_minibatch_terminal + train_minibatch_non_terminal

        batch_prev_processed_states = torch.stack(tuple(x['prev_processed_state'] for x in train_minibatch))
        prev_action_indices = [x['prev_action_index'] for x in train_minibatch]
        next_processed_state_non_terminal = torch.stack(tuple(x['next_processed_state'] for x in train_minibatch_non_terminal))

        with torch.no_grad():
            Qpred_non_terminal = self.target_Q(next_processed_state_non_terminal)
            best_vals, _ = Qpred_non_terminal.topk(1)

        rewards = (torch.tensor([x['reward'] for x in train_minibatch]).to(torch.float) )
        next_state_q = torch.cat((torch.tensor(np.zeros((len(train_minibatch_terminal),1))).to(torch.float), best_vals)).flatten()
        y_target = rewards + next_state_q

        self.Q.zero_grad()
        all_predicted_values = self.Q(batch_prev_processed_states)
        predicted_values = all_predicted_values[range(len(all_predicted_values)), prev_action_indices]
        loss = ((y_target - predicted_values)**2).sum()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())
        if self.loss_ema is None:
            self.loss_ema = loss.item()
        else:
            self.loss_ema = self.loss_ema_param * self.loss_ema + (1 - self.loss_ema_param) * loss.item()
        self.loss_ema_history.append(self.loss_ema)

        self.n_iters += 1


class Q(nn.Module):
    def __init__(self, game, hidden_size, activation_fn):
        super().__init__()
        self.game = game
        if not hasattr(hidden_size, '__len__'):
            hidden_size = (hidden_size,)
        self.hidden_size = hidden_size
        self.n_layers = len(hidden_size)

        layers = [nn.Linear(game.STATE_SIZE + 1, self.hidden_size[0])]
        for i in range(1, self.n_layers):
            layers.append(activation_fn())
            layers.append(nn.Linear(self.hidden_size[i-1], self.hidden_size[i]))

        layers.append(activation_fn())
        layers.append(nn.Linear(self.hidden_size[-1], len(self.game.ALL_ACTIONS)))

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, processed_state):
        pred_scores = self.network(processed_state)
        return pred_scores

    @staticmethod
    def process_state(state, player_index):
        return torch.tensor(np.append(state.flatten(), player_index)).to(torch.float)
