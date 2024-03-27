import enum
from dataclasses import dataclass
from .agent import AgentMulti
from typing import Type, List, Tuple
import torch
from ..game.game_multi import GameMulti
from ..mcts.mcts import MCTS, MCTSHyperparameters
import json_fix


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


class Trajectory:
    def __init__(self, states, rewards, pis, search_values=None):
        self.states = states
        self.rewards = rewards
        self.pis = pis
        self.search_values = search_values  # the mcts value for a player state or network value for env state
        self.priorities = None  # Support for prioritized sampling

    def to(self, device):
        self.states = self.states.to(device)
        self.rewards = self.rewards.to(device)
        self.pis = self.pis.to(device)
        self.search_values = self.search_values.to(device) if self.search_values is not None else None
        self.priorities = self.priorities.to(device) if self.priorities is not None else None


class AlphaAgentMultiListener:
    def after_mcts_search(self, mcts, pi, chosen_actions):
        pass

    def on_trajectories(self, trajectories):
        pass

    def advise_incoming_data_size(self, data_size):
        """
        Can be used to anticipate size of incoming data from the current game, such as to remove old data from memory
        """
        pass

    def on_data_point(self, state, pi, reward, *args, **kwargs):
        pass


class AlphaAgentDebugListener(AlphaAgentMultiListener):
    def after_mcts_search(self, mcts, pi, chosen_actions):
        import pdb
        pdb.set_trace()


class TrainingTau:
    def __init__(self, fixed_tau_value=None, tau_schedule_list=None, tau_schedule_function=None):
        if sum(map(int, [fixed_tau_value is not None, tau_schedule_list is not None,
                         tau_schedule_function is not None])) != 1:
            raise ValueError('Please pass exactly one of fixed_tau_value, tau_schedule_list, tau_schedule_function')

        self.fixed_tau_value = float(fixed_tau_value) if fixed_tau_value else None
        self.tau_schedule_list = tau_schedule_list
        self.tau_schedule_function = tau_schedule_function

        self.metrics = {}

    def get_tau(self, move_number):
        if self.fixed_tau_value is not None:
            return self.fixed_tau_value
        elif self.tau_schedule_list is not None:
            return self.tau_schedule_list[min(len(self.tau_schedule_list) - 1, move_number)]
        else:
            return self.tau_schedule_function(move_number, self.metrics)

    def update_metric(self, key, val):
        self.metrics[key] = val

    def __json__(self):
        return self.__dict__


class ValueTargetCalculationMethod(enum.Enum):
    DISCOUNTED_REWARDS = 'discounted_rewards'
    TD = 'td'
    TD_MCTS = 'td_mcts'
    TD_MCTS_NETWORK_FALLBACK = 'td_mcts_network_fallback'

    def __json__(self):
        return self.value


class TDLambda:
    def __init__(self, td_lambda):
        self.td_lambda = float(td_lambda)

    def get_lambda(self):
        return self.td_lambda

    def update_self_play_round(self, self_play_round):
        pass

    def __json__(self):
        return self.td_lambda


class TDLambdaByRound(TDLambda):
    def __init__(self, td_lambda_schedule_list):
        super().__init__(-1)
        self.td_lambda_schedule_list = td_lambda_schedule_list
        self.self_play_round = 0

    def get_lambda(self):
        return self.td_lambda_schedule_list[min(len(self.td_lambda_schedule_list) - 1, self.self_play_round)]

    def update_self_play_round(self, self_play_round):
        self.self_play_round = self_play_round

    def __json__(self):
        return self.td_lambda_schedule_list


class MCTSItersSchedule:
    def get_n_mcts_iters(self) -> Tuple[int, bool]:
        """

        :return: The number of MCTS iters to do and whether to do use the results to train the policy
        """
        raise NotImplementedError()

    def update_value(self, key, val):
        pass

    def __json__(self):
        return self.__dict__


class ConstantMCTSIters(MCTSItersSchedule):
    def __init__(self, n_mcts_iters):
        self.n_mcts_iters = int(n_mcts_iters)

    def get_n_mcts_iters(self):
        return self.n_mcts_iters, True


class FullFastMCTSIters(MCTSItersSchedule):
    """
    With some probability use a different number of MCTS iters
    """
    def __init__(self, full_mcts_iters, fast_mcts_iters, full_mcts_prob):
        self.full_mcts_iters = full_mcts_iters
        self.fast_mcts_iters = fast_mcts_iters
        self.full_mcts_prob = full_mcts_prob
        self.training = True

    def update_value(self, key, val):
        if key == 'training':
            self.training = val

    def get_n_mcts_iters(self):
        if (self.training  # Only do fast mcts in training mode; in eval mode, do the full mcts at every move
                and self.full_mcts_prob < 1.0
                and torch.rand(1).item() > self.full_mcts_prob):
            return self.fast_mcts_iters, False
        else:
            return self.full_mcts_iters, True


@dataclass(kw_only=True, slots=True)
class AlphaAgentHyperparametersMulti(MCTSHyperparameters):
    training_tau: TrainingTau = TrainingTau(1.0)
    n_mcts_iters: MCTSItersSchedule = ConstantMCTSIters(100)
    use_dirichlet_noise_in_eval: bool = False  # the AlphaGo paper is unclear about this -> defintely should be False just think about it, don't need no stinking paper
    reuse_mcts_tree: bool = False  # Initial testing showed this was actually slower (on G2048Multi) :/
    td_lambda: TDLambda = TDLambda(0.5)  # only used if value_target_calculation_method is TD. Setting this to 1 is equivalent to discounted rewards
    num_moves_td_lambda: TDLambda = TDLambda(0.5)  # used by the adaptive version


class EpisodeHistory:
    def __init__(self, n_parallel_games, state_shape, n_players, n_actions, device):
        # Initialize with empty tensors
        self.states = torch.zeros((0, n_parallel_games, *state_shape), device=device, dtype=torch.float32)
        self.rewards = torch.zeros((0, n_parallel_games, n_players), dtype=torch.float32, device=device)
        self.search_values = torch.zeros((0, n_parallel_games, n_players), dtype=torch.float32, device=device)
        self.pis = torch.zeros((0, n_parallel_games, n_actions), dtype=torch.float32, device=device)
        self.mask = torch.zeros((0, n_parallel_games), dtype=torch.bool, device=device)
        self.next_state_idx = torch.zeros(n_parallel_games, dtype=torch.long, device=device)

    def add_data(self, states, pis, mask, search_values):
        d = self.states.device
        if (self.next_state_idx[mask] >= self.states.shape[0]).any():
            blank_states = torch.zeros((1, *self.states.shape[1:]), device=d, dtype=torch.float32)
            self.states = torch.cat((self.states, blank_states))
            self.pis = torch.cat((self.pis, torch.zeros((1, *self.pis.shape[1:]), device=d, dtype=torch.float32)))
            self.rewards = torch.cat((self.rewards, torch.zeros((1, *self.rewards.shape[1:]), device=d, dtype=torch.float32)))
            self.search_values = torch.cat((self.search_values,
                                            torch.nan * torch.ones((1, *self.search_values.shape[1:]), device=d, dtype=torch.float32)))
            self.mask = torch.cat((self.mask, torch.zeros((1, *self.mask.shape[1:]), device=d, dtype=torch.bool)))

        next_state_idx = self.next_state_idx[mask]
        self.states[next_state_idx, mask] = states.to(d)

        if pis is not None:
            self.pis[next_state_idx, mask] = pis.to(d)
        else:
            self.pis[next_state_idx, mask] = torch.nan

        if search_values is not None:
            self.search_values[next_state_idx, mask] = search_values.to(d)

        self.mask[next_state_idx, mask] = True
        self.next_state_idx[mask] += 1

    def add_rewards(self, rewards, mask):
        self.rewards[self.next_state_idx[mask]-1, mask] += rewards

    def get_trajectories(self) -> List[Trajectory]:
        """
        Get the trajectories from the episode history. This is the data that will be used to train the network.

        Note, for memory efficiency after calling this method, the episode history will be empty.

        :return: A list of Trajectory objects
        """
        self.states = self.states.transpose(0, 1)
        self.rewards = self.rewards.transpose(0, 1)
        self.mask = self.mask.transpose(0, 1)
        self.pis = self.pis.transpose(0, 1)
        self.search_values = self.search_values.transpose(0, 1)

        trajectories = []

        while self.states.shape[0] > 0:
            cur_states = self.states[0]
            self.states = self.states[1:]

            cur_rewards = self.rewards[0]
            self.rewards = self.rewards[1:]

            cur_mask = self.mask[0]
            self.mask = self.mask[1:]

            cur_pis = self.pis[0]
            self.pis = self.pis[1:]

            cur_search_values = self.search_values[0]
            self.search_values = self.search_values[1:]

            trajectories.append(Trajectory(cur_states[cur_mask], cur_rewards[cur_mask], cur_pis[cur_mask],
                                           cur_search_values[cur_mask]))

            if self.states.device != 'cpu':
                torch.cuda.empty_cache()

        return trajectories

    def to(self, device):
        self.states = self.states.to(device)
        self.rewards = self.rewards.to(device)
        self.search_values = self.search_values.to(device)
        self.pis = self.pis.to(device)
        self.mask = self.mask.to(device)
        self.next_state_idx = self.next_state_idx.to(device)

    def num_data_points(self):
        return self.mask.sum().item()


# noinspection PyBroadException
class AlphaAgentMulti(AgentMulti):
    """
    This version of the alpha agent works with multiple games being played at once (so we can use the faster
    version of MCTS). It's root parallelization.
    """
    def __init__(self, game_class: Type[GameMulti], evaluator: BaseAlphaEvaluator,
                 hyperparams: AlphaAgentHyperparametersMulti, listeners: List[AlphaAgentMultiListener] = None):
        super().__init__()
        self.game = None
        self.game_class = game_class
        self.hyperparams = hyperparams
        self.evaluator = evaluator
        self.mcts = None
        self.training = True
        self.episode_history = EpisodeHistory(0, game_class.get_state_shape(),
                                              game_class.get_n_players(), game_class.get_n_actions(), 'cpu')
        self.move_number_in_current_game = 0
        self.n_players = 0
        self.listeners = listeners if listeners is not None else []

    def get_actions(self, states, mask):
        n_mcts_iters, use_to_train_network = self.hyperparams.n_mcts_iters.get_n_mcts_iters()
        self.setup_mcts(self.hyperparams, n_mcts_iters, states)
        self.mcts.search_for_n_iters(n_mcts_iters)

        tau = self.get_tau()

        if n_mcts_iters > 1:
            action_distribution = self.mcts.n[:, 1]
        else:
            self.mcts.expand()
            action_distribution = self.mcts.pi[:, 1]

        actions, pi = self.get_actions_and_pi(action_distribution, states, tau)
        mcts_value = self.mcts.w[:, 1].sum(dim=1) / self.mcts.n[:, 1].sum(dim=1).unsqueeze(-1)
        network_value = self.mcts.values[:, 1]

        for listener in self.listeners:
            listener.after_mcts_search(self.mcts, pi, actions)

        if self.training:
            if not use_to_train_network:
                # If doing fast MCTS, do not use the pi or mcts_value to train
                pi *= torch.nan
                mcts_value = None

            self.episode_history.add_data(states, pi, mask, mcts_value)

            for listener in self.listeners:
                listener.advise_incoming_data_size(self.episode_history.num_data_points())

        return actions

    def get_actions_and_pi(self, action_distribution, states, tau):
        n_parallel_games = states.shape[0]
        pi = torch.zeros((n_parallel_games, self.game_class.get_n_actions()), device=states.device, dtype=torch.float32)
        actions = torch.zeros(n_parallel_games, dtype=torch.long, device=states.device)

        if isinstance(tau, float):
            tau = torch.tensor([[tau]], device=states.device, dtype=torch.float32).repeat(n_parallel_games, 1)

        non_zero = (tau > 0).flatten()
        pi[non_zero] = action_distribution[non_zero] ** (1. / tau[non_zero])
        pi[non_zero] /= pi[non_zero].sum(dim=1, keepdim=True)
        # Choose the action according to the distribution pi
        actions[non_zero] = torch.multinomial(pi[non_zero], num_samples=1).flatten()

        pi[torch.arange(n_parallel_games, device=non_zero.device)[~non_zero], action_distribution[~non_zero].argmax(dim=1)] = 1
        actions[~non_zero] = action_distribution[~non_zero].argmax(dim=1).flatten()

        return actions, pi

    def get_tau(self):
        if self.training:
            tau = self.hyperparams.training_tau.get_tau(self.move_number_in_current_game)
        else:
            tau = 0.0

        return tau

    def setup_mcts(self, mcts_hypers, n_iters, states):
        if self.hyperparams.reuse_mcts_tree and self.mcts is not None:
            self.mcts = self.mcts.get_next_mcts(states)
        else:
            self.mcts = MCTS(self.game, self.evaluator, mcts_hypers, n_iters, states,
                             add_dirichlet_noise=(self.training or self.hyperparams.use_dirichlet_noise_in_eval))

    def on_rewards(self, rewards: torch.Tensor, mask: torch.Tensor):
        if self.training:
            self.episode_history.add_rewards(rewards, mask)

    def before_env_move(self, states: torch.Tensor, mask: torch.Tensor):
        if self.training:
            # Put this in the episode history because we're gonna learn the value for these states. pi will be nans
            self.episode_history.add_data(
                states, None,
                mask, None
            )

            for listener in self.listeners:
                listener.advise_incoming_data_size(self.episode_history.num_data_points())

    def train(self):
        self.training = True
        self.hyperparams.n_mcts_iters.update_value('training', True)

    def eval(self):
        self.training = False
        self.hyperparams.n_mcts_iters.update_value('training', False)

    def before_game_start(self, game: GameMulti):
        self.game = game
        self.episode_history = EpisodeHistory(game.n_parallel_games, self.game_class.get_state_shape(),
                                              self.game_class.get_n_players(), self.game_class.get_n_actions(),
                                              game.states.device)
        self.move_number_in_current_game = 0

        if game.states.device != 'cpu':
            torch.cuda.empty_cache()

    def on_game_restart(self, game):
        self.game = game

    def on_game_end(self):
        if not self.training:
            return

        trajectories = self.get_trajectories()

        for data_listener in self.listeners:
            data_listener.on_trajectories(trajectories)

    def to(self, device):
        """
        Move the agent's data to the specified device

        :param device:
        """
        self.episode_history.to(device)

    def get_trajectories(self):
        return self.episode_history.get_trajectories()


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
