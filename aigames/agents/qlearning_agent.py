import copy
from collections import defaultdict
import numpy as np
import torch
from aigames.base.agent import *
from aigames.base.utils import *


class QLearningAgent(SequentialAgent):
    def __init__(self, game, Q, exploration_probability=0.1,
                 batch_size=16, lr=0.01, update_target_Q_every=10000,
                 min_replay_memory_size=10000, max_replay_memory_size=10000,
                 device='cpu'):
        """

        :param Q: the Q predictor to be used
        """
        super().__init__(game)

        self.game = game
        self.exploration_probability = exploration_probability
        self.training = True

        self.device = torch.device(device)
        self.Q = Q
        self.Q.to(self.device)
        self.target_Q = copy.deepcopy(Q)
        self.target_Q.to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr = lr)

        self.replay_memory = QLearningAgentReplayMemory(max_replay_memory_size)
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
        self.prev_processed_state = {}
        self.prev_rewards = defaultdict(int)
        self._all_processed_state_actions = {}

    def choose_action(self, state, player_index, verbose = False):
        legal_actions = self.game.legal_actions(state)
        legal_action_indices = [self.game.ALL_ACTIONS.index(a) for a in legal_actions]

        # epsilon-Greedy
        if (self.training and np.random.random() < self.exploration_probability) or len(self.replay_memory) < self.min_replay_memory_size:
            # Explore
            idx = np.random.choice(legal_action_indices)
        else:
            # Exploit
            self.Q.eval()
            all_processed_state_actions = self.get_all_processed_state_actions(state, player_index)
            all_processed_state_actions = strip_nans(all_processed_state_actions).to(self.device)
            with torch.no_grad():
                scores = self.Q(all_processed_state_actions)

            idx = legal_action_indices[np.argmax(scores)]

        action = self.game.ALL_ACTIONS[idx]
        self.prev_processed_state[player_index] = self.Q.process_state_action(state, action,player_index)
        return action

    def get_all_processed_state_actions(self, state, player_index):
        return self._get_all_processed_state_actions(self.game, state, player_index, self.Q)

    def _get_all_processed_state_actions(self, game, state, player_index, Q):
        key = None
        if hasattr(game, 'hashable_state'):
            key = (game, game.hashable_state(state), player_index, Q)
            if key in self._all_processed_state_actions:
                return self._all_processed_state_actions[key]

        legal_actions = game.legal_actions(state)
        out = torch.cat(tuple(Q.process_state_action(state, action, player_index) for action in legal_actions), 0)
        n_nans_to_add = len(game.ALL_ACTIONS) - out.shape[0]
        nans = torch.FloatTensor(np.nan * np.ones((n_nans_to_add, *out.shape[1:])))
        out = torch.cat((out, nans), 0)

        if key is not None:
            self._all_processed_state_actions[key] = out

        return out

    def reward(self, reward_value, next_state, player_index):
        # Don't do anything if this is just the initial reward
        if player_index not in self.prev_processed_state or not self.training:
            return

        if player_index != self.game.get_player_index(next_state) and not self.game.is_terminal_state(next_state):
            # just add this reward to be included when we get back to this player
            self.prev_rewards[player_index] += reward_value
            return

        terminal = self.game.is_terminal_state(next_state)
        all_processed_next_state_actions = None
        if not terminal:
            all_processed_next_state_actions = self.get_all_processed_state_actions(next_state, player_index)

        self.replay_memory.add(terminal, self.prev_processed_state[player_index], self.prev_rewards[player_index] + reward_value,
                               all_processed_next_state_actions)

        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        if (self.n_iters + 1) % self.update_target_Q_every == 0:
            self.target_Q = copy.deepcopy(self.Q)

        # Sample a minibatch and train
        terminal_minibatch, nonterminal_minibatch = self.replay_memory.sample(self.batch_size)

        processed_state_actions = torch.FloatTensor().to(self.device)
        rewards = torch.FloatTensor().to(self.device)
        next_state_q = torch.FloatTensor().to(self.device)

        if terminal_minibatch is not None:
            terminal_processed_state_actions, terminal_rewards = map(lambda x: x.to(self.device), terminal_minibatch)
            processed_state_actions = torch.cat((processed_state_actions, terminal_processed_state_actions), 0)
            rewards = torch.cat((rewards, terminal_rewards), 0)
            next_state_q = torch.cat(
                (next_state_q,
                 torch.FloatTensor(np.zeros((len(terminal_rewards), 1))).to(self.device)
                 ),
                0
            )

        if nonterminal_minibatch is not None:
            nonterminal_processed_state_actions, nonterminal_rewards, nonterminal_all_processed_next_state_actions = (
                map(lambda x: x.to(self.device), nonterminal_minibatch))

            with torch.no_grad():
                best_vals = torch.cat(
                    tuple(
                        torch.FloatTensor([self.target_Q(strip_nans(x).to(self.device)).max()])
                        for x in nonterminal_all_processed_next_state_actions
                    )
                ).unsqueeze(1)

            processed_state_actions = torch.cat((processed_state_actions, nonterminal_processed_state_actions), 0)
            rewards = torch.cat((rewards, nonterminal_rewards), 0)
            next_state_q = torch.cat((next_state_q, best_vals), 0)

        y_target = rewards + next_state_q
        self.Q.train()
        self.Q.zero_grad()
        predicted_values = self.Q(processed_state_actions)
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

    def eval(self):
        self.training = False
        self.Q.eval()

    def train(self):
        self.training = True
        self.Q.train()


class QLearningAgentReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.terminal_processed_state_actions = torch.FloatTensor(())
        self.terminal_rewards = torch.FloatTensor(())
        self.nonterminal_processed_state_actions = torch.FloatTensor(())
        self.nonterminal_rewards = torch.FloatTensor(())
        self.nonterminal_all_processed_next_state_actions = torch.FloatTensor(())
        self.terminal_states = []

    def add(self, next_state_is_terminal, processed_state_action, reward,
            all_processed_next_state_actions=None):
        while len(self) >= self.max_size:
            self.pop()

        if next_state_is_terminal:
            self.terminal_states.append(True)
            self.terminal_processed_state_actions = torch.cat(
                (self.terminal_processed_state_actions, processed_state_action),
                0
            )
            self.terminal_rewards = torch.cat(
                (self.terminal_rewards, torch.FloatTensor([reward]).unsqueeze(0)),
                0
            )
        else:
            self.terminal_states.append(False)
            self.nonterminal_processed_state_actions = torch.cat(
                (self.nonterminal_processed_state_actions, processed_state_action),
                0
            )
            self.nonterminal_rewards = torch.cat(
                (self.nonterminal_rewards, torch.FloatTensor([reward]).unsqueeze(0)),
                0
            )
            self.nonterminal_all_processed_next_state_actions = torch.cat(
                (self.nonterminal_all_processed_next_state_actions, all_processed_next_state_actions.unsqueeze(0)),
                0
            )

    def __len__(self):
        return len(self.terminal_processed_state_actions) + len(self.nonterminal_processed_state_actions)

    def pop(self):
        terminal = self.terminal_states.pop(0)
        if terminal:
            self.terminal_processed_state_actions = self.terminal_processed_state_actions[1:]
            self.terminal_rewards = self.terminal_rewards[1:]
        else:
            self.nonterminal_processed_state_actions = self.nonterminal_processed_state_actions[1:]
            self.nonterminal_rewards = self.nonterminal_rewards[1:]
            self.nonterminal_all_processed_next_state_actions = self.nonterminal_all_processed_next_state_actions[1:]

    def sample(self, batch_size):
        frac_terminal = len(self.terminal_processed_state_actions) / float(len(self))
        n_terminal = np.random.binomial(batch_size, frac_terminal)

        terminal_dataset = torch.utils.data.TensorDataset(self.terminal_processed_state_actions, self.terminal_rewards)
        nonterminal_dataset = torch.utils.data.TensorDataset(self.nonterminal_processed_state_actions,
                                                             self.nonterminal_rewards,
                                                             self.nonterminal_all_processed_next_state_actions)

        if n_terminal == 0:
            nonterminal_dataloader = torch.utils.data.DataLoader(nonterminal_dataset, batch_size=(batch_size - n_terminal),
                                                              shuffle=True)
            return None, next(iter(nonterminal_dataloader))
        elif n_terminal == batch_size:
            terminal_dataloader = torch.utils.data.DataLoader(terminal_dataset, batch_size=n_terminal, shuffle=True)
            return next(iter(terminal_dataloader)), None
        else:
            terminal_dataloader = torch.utils.data.DataLoader(terminal_dataset, batch_size=n_terminal, shuffle=True)
            nonterminal_dataloader = torch.utils.data.DataLoader(nonterminal_dataset, batch_size=(batch_size - n_terminal),
                                                              shuffle=True)
            return next(iter(terminal_dataloader)), next(iter(nonterminal_dataloader))
