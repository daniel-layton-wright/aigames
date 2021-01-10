from ..agent.qlearning_agent import *
from ..game import *
import torch
import torch.utils.data
import copy
from typing import Type
import torch.nn as nn
import time
from tqdm.auto import tqdm
import multiprocessing as mp


def is_json_serializable(x):
    import json
    try:
        json.dumps(x)
        return True
    except:
        return False


class TrainingListener(GameListener):
    def before_begin_training(self, training_manager):
        pass

    def on_training_step(self, iter: int, loss: float, training_manager, **kwargs):
        pass


class QLearningTrainingRun(GameListener):
    def __init__(self, game: Type[PartiallyObservableSequentialGame], network, optimizer, state_shape,
                 exploration_probability=0.1, discount_rate=0.99, update_target_network_ever_n_iters=1,
                 min_data_size=1000, max_data_size=50000, batch_size=32, share_among_players=True,
                 frac_terminal_to_sample=None, training_listeners: List[TrainingListener] = (),
                 sample_proportional_to_reward=False, sampling_baseline_reward=None):
        """

        :param game:
        :param network: Pass in a callable that returns a network
        :param optimizer: Pass in a function that will do f( network ) -> optimizer
        :param state_shape:
        :param exploration_probability:
        :param discount_rate:
        :param update_target_network_ever_n_iters:
        :param min_data_size:
        :param max_data_size:
        :param batch_size:
        :param share_among_players:
        :param training_listeners:
        """
        self.game = game
        self.networks = []
        self.optimizers = []
        self.q_optimizers = []
        self.state_shape = state_shape
        self.q_functions = []
        self.target_networks = []
        self.datasets = []
        self.listeners = training_listeners

        self.exploration_probability = exploration_probability
        self.discount_rate = discount_rate
        self.min_data_size = min_data_size
        self.batch_size = batch_size
        self.discount_rate = discount_rate

        self.q_optimizers = []

        self.agents = []
        for i in range(self.game.get_n_players()):
            if i == 0 or (not share_among_players):
                self.networks.append( network() )
                self.optimizers.append( optimizer(self.networks[-1]) )
                self.datasets.append( BasicQLearningDataset(self.game, self.state_shape, max_size=max_data_size) )
                self.target_networks.append( copy.deepcopy(self.networks[-1]) )
                self.q_optimizers.append(
                    QLearningOptimizer(self.datasets[-1], self.optimizers[-1], self.networks[-1],
                                       self.target_networks[-1], batch_size=self.batch_size,
                                       discount_rate=self.discount_rate, frac_terminal_to_sample=frac_terminal_to_sample,
                                       sample_proportional_to_reward=sample_proportional_to_reward,
                                       sampling_baseline_reward=sampling_baseline_reward)
                )
                self.q_functions.append( BasicQLearningFunction(self.networks[-1], self.state_shape) )

            cur_agent = QLearningAgent(self.game, i, self.q_functions[-1], data_listener=self.datasets[-1],
                                       exploration_probability_scheduler=ConstantExplorationProbabilityScheduler(self.exploration_probability),
                                       discount_rate=self.discount_rate)
            self.agents.append(cur_agent)

        self.update_target_network_ever_n_iters = update_target_network_ever_n_iters
        self.n_iters = 0

        for listener in self.listeners:
            listener.before_begin_training(self)

    def train(self, n_games=10000):
        game = self.game(self.agents, [self, *self.listeners])
        for _ in tqdm(range(n_games)):
            game.play()

    def take_training_step(self):
        for i, (dataset, q_optimizer) in enumerate(zip(self.datasets, self.q_optimizers)):
            if len(dataset) < self.min_data_size:
                return

            loss = q_optimizer.take_training_step()

            if (self.n_iters % self.update_target_network_ever_n_iters) == 0:
                q_optimizer.reset_target_network()

            if i == 0:  # TODO : fix this hack (only calling back to listener for first network)
                for listener in self.listeners:
                    listener.on_training_step(self.n_iters, loss, self)

        self.n_iters += 1

    def on_action(self, game, action):
        if any(len(dataset) > 0 for dataset in self.datasets):
            self.take_training_step()

    def play_debug_game(self):
        cli = CommandLineGame(clear_screen=False)
        debugger = DebugGameListener()

        for agent in self.agents:
            agent.eval()

        game = self.game(self.agents, listeners=[cli, debugger])
        game.play()


class DebugGameListener(GameListener):
    def before_action(self, game: SequentialGame, legal_actions):
        all_actions = game.get_all_actions()
        legal_action_indices = [all_actions.index(legal_action) for legal_action in legal_actions]
        cur_player_index = game.get_cur_player_index(game.state)
        with torch.no_grad():
            if 'Q' in game.players[cur_player_index].__dict__:
                print(game.players[cur_player_index].Q.evaluate(game.state)[legal_action_indices])


def take_training_step(terminal_minibatch, nonterminal_minibatch, target_network, network, optimizer,
                       discount_rate, device):
    terminal_states, terminal_action_indices, terminal_rewards = terminal_minibatch
    nonterminal_states, nonterminal_action_indices, nonterminal_rewards, nonterminal_next_states, nonterminal_next_state_legal_action_maps = nonterminal_minibatch

    # Compute loss
    if len(nonterminal_next_states) > 0:
        with torch.no_grad():
            network_result = (target_network(nonterminal_next_states.to(device)) + nonterminal_next_state_legal_action_maps).max(dim=1)[0]

        nonterminal_y = nonterminal_rewards.squeeze(1) + discount_rate * network_result
    else:
        nonterminal_y = torch.Tensor([])

    optimizer.zero_grad()
    # print(terminal_states)
    # print(terminal_action_indices)
    # print(terminal_rewards)

    action_indices = torch.cat((nonterminal_action_indices.squeeze(1), terminal_action_indices.squeeze(1)))
    y_pred = network(torch.cat((nonterminal_states, terminal_states)).to(device))[range(len(action_indices)), action_indices]
    y = torch.cat((nonterminal_y, terminal_rewards.squeeze(1)))

    per_point_nonterminal_loss = ((y_pred[:len(nonterminal_states)] - y[:len(nonterminal_states)]) ** 2)
    nonterminal_loss = per_point_nonterminal_loss.mean()
    per_point_terminal_loss = ((y_pred[len(nonterminal_states):] - y[len(nonterminal_states):]) ** 2)
    terminal_loss = per_point_terminal_loss.mean()
    per_point_loss = ((y_pred - y) ** 2)
    loss = per_point_loss.mean()

    # really_wrong = (per_point_terminal_loss > 0.5)
    # if sum(really_wrong) > 0:
    #     print(terminal_states[really_wrong])
    #     print(terminal_action_indices[really_wrong])
    #     print(terminal_rewards[really_wrong])
    #     print(y_pred[len(nonterminal_states):][really_wrong])
    #     print('\n\n\n')
    #
    # really_wrong = (per_point_nonterminal_loss > 0.5)
    # if sum(really_wrong) > 0:
    #     print(nonterminal_states[really_wrong])
    #     print(nonterminal_next_states[really_wrong])
    #     print(network_result[really_wrong])
    #     print(nonterminal_rewards[really_wrong])
    #     print(nonterminal_y[really_wrong])
    #     print(y_pred[:len(nonterminal_states)][really_wrong])
    #     print('\n\n\n')
    #
    # if len(nonterminal_states) > 0:
    #     print(nonterminal_states[0])
    #     print(nonterminal_action_indices[0])
    #     print(nonterminal_rewards[0])
    #     print(network_result[0])
    #     print(y[0])
    #     print(y_pred[0])
    #     print(y)
    #     print(y_pred)
    #     print(nonterminal_loss)
    #     print('\n\n\n')

    # Back-prop
    loss.backward()

    # Optimization step
    optimizer.step()

    return loss.item(), terminal_loss.item(), nonterminal_loss.item()


class QLearningOptimizer:
    def __init__(self, dataset, optimizer, network: nn.Module, target_network, device=torch.device('cpu'),
                 batch_size=32, discount_rate=0.99, frac_terminal_to_sample=None, sample_proportional_to_reward=False,
                 sampling_baseline_reward=None):
        self.dataset = dataset
        self.optimizer = optimizer
        self.network = network
        self.target_network = target_network
        self.device = device
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.frac_terminal_to_sample = frac_terminal_to_sample
        self.sample_proportional_to_reward = sample_proportional_to_reward
        self.sampling_baseline_reward = sampling_baseline_reward

    def take_training_step(self):
        terminal_minibatch, nonterminal_minibatch = self.dataset.sample_minibatch(self.batch_size,
                                                                                  frac_terminal=self.frac_terminal_to_sample,
                                                                                  sample_proportional_to_reward=self.sample_proportional_to_reward,
                                                                                  baseline_reward=self.sampling_baseline_reward)

        loss, terminal_loss, nonterminal_loss = take_training_step(terminal_minibatch, nonterminal_minibatch,
                                                                   self.target_network, self.network,
                                                                   self.optimizer, self.discount_rate, self.device)
        return loss

    def reset_target_network(self):
        self.target_network = copy.deepcopy(self.network)


class BasicQLearningFunction(QLearningFunction):
    def __init__(self, network: nn.Module, state_shape, device=torch.device('cpu')):
        self.network = network
        self.state_shape = state_shape
        self.device = device

    def evaluate(self, state):
        with torch.no_grad():
            return self.network(torch.Tensor(state).reshape(self.state_shape).unsqueeze(0).to(self.device)).squeeze()


class ListDataset(torch.utils.data.Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(l) for l in lists)
        self.lists = lists

    def __getitem__(self, index):
        return tuple(lists[index] for lists in self.lists)

    def __len__(self):
        return len(self.lists[0])


class BasicQLearningDataset(QLearningDataListener):
    def __init__(self, game: Type[PartiallyObservableSequentialGame], state_shape, max_size=np.inf,
                 debug_mode=False):
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

        self.max_size = max_size
        self.debug_mode = debug_mode

    def on_SARS(self, state, action_index, reward, next_state):
        if self.debug_mode and len(self) >= self.max_size:
            return

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

        while len(self) > self.max_size:
            self.pop()

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

    def sample_minibatch(self, batch_size, frac_terminal=None, sample_proportional_to_reward=False, baseline_reward=1):
        if sample_proportional_to_reward:
            terminal_reward_weights = self._get_reward_weights(self.terminal_rewards, baseline_reward)
            nonterminal_reward_weights = self._get_reward_weights(self.nonterminal_rewards, baseline_reward)
            frac_terminal = sum(terminal_reward_weights) / (sum(terminal_reward_weights) + sum(nonterminal_reward_weights))
            frac_terminal = frac_terminal[0]
        elif frac_terminal is None:
            frac_terminal = len(self.terminal_actions) / float(len(self))

        n_terminal = int(np.random.binomial(batch_size, frac_terminal))

        terminal_data = (torch.empty(0, *self.state_shape), torch.empty(0, 1, dtype=torch.int64), torch.empty(0, 1))
        nonterminal_data = (torch.empty(0, *self.state_shape), torch.empty(0, 1, dtype=torch.int64), torch.empty(0, 1),
                            torch.empty(0, *self.state_shape), torch.empty(0, self.n_actions))

        terminal_dataset = ListDataset(self.terminal_states, self.terminal_actions, self.terminal_rewards)
        nonterminal_dataset = ListDataset(self.nonterminal_states, self.nonterminal_actions, self.nonterminal_rewards,
                                          self.nonterminal_next_states, self.nonterminal_next_state_legal_action_maps)

        if n_terminal > 0:
            if sample_proportional_to_reward:
                sampler = torch.utils.data.WeightedRandomSampler(terminal_reward_weights, batch_size, replacement=True)
            else:
                sampler = torch.utils.data.RandomSampler(terminal_dataset, replacement=True, num_samples=n_terminal)
            terminal_dataloader = torch.utils.data.DataLoader(terminal_dataset, sampler=sampler, batch_size=n_terminal)
            terminal_data = next(iter(terminal_dataloader))

        if (batch_size - n_terminal) > 0:
            if sample_proportional_to_reward:
                sampler = torch.utils.data.WeightedRandomSampler(nonterminal_reward_weights, batch_size - n_terminal,
                                                                 replacement=True)
            else:
                sampler = torch.utils.data.RandomSampler(nonterminal_dataset, replacement=True,
                                                         num_samples=(batch_size - n_terminal))
            nonterminal_dataloader = torch.utils.data.DataLoader(nonterminal_dataset, sampler=sampler,
                                                                 batch_size=(batch_size - n_terminal))
            nonterminal_data = next(iter(nonterminal_dataloader))

        return terminal_data, nonterminal_data

    @staticmethod
    def _get_reward_weights(list_of_rewards, baseline_reward):
        min_reward = min(list_of_rewards)
        reward_weights = list(map(lambda x: x.item() + min_reward + baseline_reward, list_of_rewards))
        return reward_weights


class ConstantExplorationProbabilityScheduler(ExplorationProbabilityScheduler):
    def __init__(self, exploration_probability):
        self.exploration_probability = exploration_probability

    def get_exploration_probability(self, agent, state):
        return self.exploration_probability

