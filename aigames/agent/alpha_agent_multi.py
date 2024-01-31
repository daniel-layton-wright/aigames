import numpy as np
from .agent import Agent, AgentMulti
from ..game.game import SequentialGame
from typing import Type, List
from tqdm.auto import tqdm
import torch

from ..game.game_multi import GameMulti
from ..mcts.mcts import MCTS, MCTSHyperparameters


class BaseAlphaEvaluator:
    def __init__(self):
        self.device = torch.device('cpu')

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
        if sum(map(int, [fixed_tau_value is not None, tau_schedule_list is not None,
                         tau_schedule_function is not None])) != 1:
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


class AlphaAgentHyperparametersMulti:
    __slots__ = ['mcts_hyperparams', 'n_mcts', 'discount_rate', 'training_tau',
                 'use_dirichlet_noise_in_eval']

    def __init__(self):
        self.mcts_hyperparams = MCTSHyperparameters()
        self.discount_rate = 0.99
        self.training_tau = TrainingTau(1.0)
        self.use_dirichlet_noise_in_eval = False  # the AlphaGo paper is unclear about this


# noinspection PyBroadException
class AlphaAgentMulti(AgentMulti):
    """
    This version of the alpha agent works with multiple games being played at once (so we can use the faster
    version of MCTS). It's root parallelization.
    """
    def __init__(self, game_class: Type[GameMulti], evaluator: BaseAlphaEvaluator,
                 hyperparams: AlphaAgentHyperparametersMulti, listeners: List[AlphaAgentListener] = None,
                 use_tqdm: bool = False):
        super().__init__()
        self.game = None
        self.game_class = game_class
        self.hyperparams = hyperparams
        self.evaluator = evaluator
        self.mcts = None
        self.training = True
        self.episode_history = []
        self.move_number_in_current_game = 0
        self.n_players = 0
        self.listeners = listeners if listeners is not None else []
        self.use_tqdm = use_tqdm

    def get_actions(self, states, is_terminal=None):
        # TODO: reuse previous nodes
        n_parallel_games = states.shape[0]
        self.hyperparams.mcts_hyperparams.n_roots = n_parallel_games
        self.mcts = MCTS(self.game, self.evaluator, self.hyperparams.mcts_hyperparams, states)

        # Add Dirichlet noise to the root node
        if self.training or self.hyperparams.use_dirichlet_noise_in_eval:
            self.mcts.add_dirichlet_noise()

        self.mcts.search_for_n_iters(self.hyperparams.mcts_hyperparams.n_iters)

        if self.training:
            tau = self.hyperparams.training_tau.get_tau(self.move_number_in_current_game)
        else:
            tau = 0.0

        if self.hyperparams.n_mcts > 1:
            action_distribution = self.mcts.n[:, 1]
        else:
            self.mcts.expand()
            action_distribution = self.mcts.pi[:, 1]

        pi = torch.zeros((n_parallel_games, self.game_class.get_n_actions()))
        actions = torch.zeros(n_parallel_games, dtype=torch.long)

        if tau > 0:
            pi = action_distribution ** (1. / tau)
            pi /= pi.sum(dim=1, keepdim=True)
            # Choose the action according to the distribution pi
            actions = torch.multinomial(pi, num_samples=1).flatten()
        else:
            pi[torch.arange(n_parallel_games), action_distribution.argmax(dim=1)] = 1
            actions = action_distribution.argmax(dim=1).flatten()

        # Record this for training, set the root node to the next node now and forget the parent
        player_index = self.game.get_cur_player_index(states)
        self.episode_history.append(TimestepData(states, pi, player_index))

        return actions

    def on_rewards(self, rewards):
        pass

    def train(self):
        self.evaluator.train()
        self.training = True

    def eval(self):
        self.evaluator.eval()
        self.training = False

    def before_game_start(self, game):
        self.game = game
        self.episode_history = []
        self.move_number_in_current_game = 0

    def on_game_end(self):
        return

        if not self.training:
            return

        episode_history = reversed(self.episode_history)  # work backwards
        cum_rewards = [0 for _ in range(self.n_players)]

        for data in episode_history:
            if isinstance(data, RewardData):
                cum_rewards[data.player_index] = (data.reward_value +
                                                  self.hyperparams.discount_rate * cum_rewards[data.player_index])
            elif isinstance(data, TimestepData):
                reward = cum_rewards[data.player_index]
                for data_listener in self.listeners:
                    data_listener.on_data_point(data.state, data.pi, reward)


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


class DummyAlphaEvaluatorMulti(BaseAlphaEvaluator):
    def __init__(self, n_actions, n_players, device='cpu'):
        super().__init__()
        self.n_actions = n_actions
        self.device = device
        self.P = torch.ones(n_actions, dtype=torch.float32, device=device)
        self.P /= self.P.sum()

        self.n_players = n_players

    def evaluate(self, states):
        return (self.P.repeat(states.shape[0]).reshape(states.shape[0], -1),
                torch.zeros((states.shape[0], self.n_players), device=self.device))

    def process_state(self, state):
        return state

    def eval(self):
        # Switch to eval mode
        pass

    def train(self):
        # Switch to train mode
        pass
