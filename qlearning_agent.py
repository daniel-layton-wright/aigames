import numpy as np
import copy
import torch
import torch.nn as nn

from agent import *


class QLearningAgent(SequentialAgent):
    def __init__(self, game, Q, exploration_probability = 0.1,
                 batch_size = 16, lr = 0.01):
        """

        :param Q: the Q predictor to be used
        """
        self.game = game
        self.exploration_probability = exploration_probability
        self.training = True

        self.Q = Q
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr = lr)
        self.replay_memory = []
        self.loss_ema = None

        # Used to save so that when the reward comes back we can add to the replay memory
        self.prev_state = None
        self.prev_processed_state = None
        self.prev_action_index = None

    def choose_action(self, state, player_index):
        processed_state = self.Q.process_state(state, player_index)

        # epsilon-Greedy
        if self.training and np.random.random() < self.exploration_probability:
            # Explore
            idx = np.random.choice(len(self.game.ALL_ACTIONS))
        else:
            # Exploit
            with torch.no_grad():
                idx = self.Q(processed_state)
                pred_scores = self.Q(processed_state)
                _, idx = pred_scores.topk(1)

        self.prev_state = copy.deepcopy(state)
        self.prev_processed_state = processed_state
        self.prev_action_index = idx
        return self.game.ALL_ACTIONS[idx]

    def reward(self, reward_value, next_state, player_index):
        # Don't do anything if this is just the initial reward
        if self.prev_state is None or not self.training:
            return

        # Add to replay memory
        self.replay_memory.append(
            {
                'prev_state': self.prev_state,
                'prev_processed_state': self.prev_processed_state,
                'prev_action_index': self.prev_action_index,
                'next_state': copy.deepcopy(next_state),
                'next_processed_state': self.Q.process_state(next_state, player_index),
                'reward': reward_value,
                'next_state_is_terminal': self.game.is_terminal_state(next_state)
            }
        )

        # Sample a minibatch and train
        train_minibatch = np.random.choice(self.replay_memory, self.batch_size)
        train_minibatch_terminal = [x for x in train_minibatch if x['next_state_is_terminal']]
        train_minibatch_non_terminal = [x for x in train_minibatch if not x['next_state_is_terminal']]
        train_minibatch = train_minibatch_terminal + train_minibatch_non_terminal

        batch_prev_processed_states = torch.stack(tuple(x['prev_processed_state'] for x in train_minibatch))
        prev_action_indices = [x['prev_action_index'] for x in train_minibatch]
        next_processed_state_non_terminal = torch.stack(tuple(x['next_processed_state'] for x in train_minibatch_non_terminal))

        with torch.no_grad():
            Qpred_non_terminal = self.Q(next_processed_state_non_terminal)
            best_vals, _ = Qpred_non_terminal.topk(1)

        rewards = (torch.tensor([x['reward'] for x in train_minibatch]).to(torch.float) )# +
        next_state_q = torch.cat((torch.tensor(np.zeros((len(train_minibatch_terminal),1))).to(torch.float), best_vals)).flatten()
        y_target = rewards + next_state_q

        self.Q.zero_grad()
        all_predicted_values = self.Q(batch_prev_processed_states)
        predicted_values = all_predicted_values[range(len(all_predicted_values)), prev_action_indices]
        loss = ((y_target - predicted_values)**2).sum()
        loss.backward()
        self.optimizer.step()

        if self.loss_ema is None:
            self.loss_ema = loss.item()
        else:
            self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss.item()


class Q(nn.Module):
    def __init__(self, game, hidden_size, activation_fn):
        super().__init__()
        self.game = game
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(game.STATE_SIZE + 1, self.hidden_size)  # Extra 1 is for the player index
        self.activation_fn = activation_fn()
        self.h2o = nn.Linear(self.hidden_size, len(game.ALL_ACTIONS))

    def forward(self, processed_state):
        pred_scores = self.h2o(self.activation_fn(self.i2h(processed_state)))
        return pred_scores

    @staticmethod
    def process_state(state, player_index):
        return torch.tensor(np.append(state.flatten(), player_index)).to(torch.float)
