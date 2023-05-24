from .qlearning_training_manager import QLearningFunction, QLearningDataListener, ListDataset
from ..game import PartiallyObservableSequentialGame, SequentialGame
import torch
import torch.nn as nn
from typing import Type
import numpy as np


class QLearningFunctionV2(QLearningFunction):
    def __init__(self, network: nn.Module, game: SequentialGame, player_index, state_shape, device=torch.device('cpu')):
        self.network = network
        self.game = game
        self.player_index = player_index
        self.all_actions = self.game.get_all_actions()
        self.n_actions = len(self.all_actions)
        self.state_shape = state_shape
        self.device = device

    def evaluate(self, state):
        legal_actions = self.game.get_legal_actions(state)
        legal_action_indices = [self.all_actions.index(a) for a in legal_actions]
        next_states_and_rewards = [self.game.get_next_state_and_rewards(state.copy(), a) for a in legal_actions]
        rewards = torch.Tensor([reward[self.player_index] for _, reward in next_states_and_rewards])
        next_states = torch.stack(tuple(torch.Tensor(next_state).reshape(self.state_shape) for next_state, _ in next_states_and_rewards))
        out = -np.inf * torch.ones(self.n_actions)

        with torch.no_grad():
            out[legal_action_indices] = rewards + self.network(next_states).to(self.device).squeeze()

        return out


class QLearningDatasetV2(QLearningDataListener):
    def __init__(self, game: Type[PartiallyObservableSequentialGame], state_shape):
        self.game = game
        self.state_shape = state_shape
        self.all_actions = self.game.get_all_actions()
        self.n_actions = len(self.all_actions)

        self.nonterminal_states = []
        self.nonterminal_actions = []
        self.nonterminal_rewards = []
        self.nonterminal_next_states = []
        self.nonterminal_next_state_legal_action_maps = []

        self.terminal_states = []
        self.terminal_actions = []
        self.terminal_rewards = []

        self.order_is_terminal = []  # Order that the data was inserted (for pop-ing)

    def on_SARS(self, state, action_index, reward, next_state):
        if self.game.is_terminal_state(next_state):
            self.terminal_states.append(torch.Tensor(state).reshape(self.state_shape))
            self.terminal_actions.append(torch.LongTensor([action_index]))
            self.terminal_rewards.append(torch.Tensor([reward]))
            self.order_is_terminal.append(True)
        else:
            self.nonterminal_states.append(torch.Tensor(state).reshape(self.state_shape))
            self.nonterminal_actions.append(torch.LongTensor([action_index]))
            self.nonterminal_rewards.append(torch.Tensor([reward]))
            self.nonterminal_next_states.append(torch.Tensor(next_state).reshape(self.state_shape))

            mask = -np.inf * torch.ones(self.n_actions)
            legal_action_indices = [self.all_actions.index(a) for a in self.game.get_legal_actions(next_state)]
            mask[legal_action_indices] = 0

            self.nonterminal_next_state_legal_action_maps.append(mask)
            self.order_is_terminal.append(False)

    def pop(self):
        pop_terminal = self.order_is_terminal.pop()
        if pop_terminal:
            self.terminal_states.pop()
            self.terminal_actions.pop()
            self.terminal_rewards.pop()
        else:
            self.nonterminal_states.pop()
            self.nonterminal_actions.pop()
            self.nonterminal_rewards.pop()
            self.nonterminal_next_states.pop()
            self.nonterminal_next_state_legal_action_maps.pop()

    def __len__(self):
        return len(self.nonterminal_states) + len(self.terminal_states)

    def sample_minibatch(self, batch_size):
        frac_terminal = len(self.terminal_actions) / float(len(self))
        n_terminal = np.random.binomial(batch_size, frac_terminal)

        terminal_data = (torch.empty(0, *self.state_shape), torch.empty(0, 1, dtype=torch.int64), torch.empty(0, 1))
        nonterminal_data = (torch.empty(0, *self.state_shape), torch.empty(0, 1, dtype=torch.int64), torch.empty(0, 1),
                            torch.empty(0, *self.state_shape), torch.empty(0, self.n_actions))

        terminal_dataset = ListDataset(self.terminal_states, self.terminal_actions, self.terminal_rewards)
        nonterminal_dataset = ListDataset(self.nonterminal_states, self.nonterminal_actions, self.nonterminal_rewards,
                                          self.nonterminal_next_states, self.nonterminal_next_state_legal_action_maps)

        if n_terminal > 0:
            sampler = torch.utils.data.RandomSampler(terminal_dataset, replacement=True, num_samples=n_terminal)
            terminal_dataloader = torch.utils.data.DataLoader(terminal_dataset, sampler=sampler, batch_size=n_terminal)
            terminal_data = next(iter(terminal_dataloader))

        if (batch_size - n_terminal) > 0:
            sampler = torch.utils.data.RandomSampler(nonterminal_dataset, replacement=True,
                                                     num_samples=(batch_size - n_terminal))
            nonterminal_dataloader = torch.utils.data.DataLoader(nonterminal_dataset, sampler=sampler,
                                                                 batch_size=(batch_size - n_terminal))
            nonterminal_data = next(iter(nonterminal_dataloader))

        return terminal_data, nonterminal_data
