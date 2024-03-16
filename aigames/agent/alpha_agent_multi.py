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


class AlphaAgentMultiListener:
    def after_mcts_search(self, mcts, pi, chosen_actions):
        pass

    def on_trajectories(self, trajectories):
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
    value_target_calculation_method: ValueTargetCalculationMethod = ValueTargetCalculationMethod.TD_MCTS_NETWORK_FALLBACK
    td_lambda: TDLambda = TDLambda(0.5)  # only used if value_target_calculation_method is TD. Setting this to 1 is equivalent to discounted rewards
    num_moves_td_lambda: TDLambda = TDLambda(0.5)  # used by the adaptive version


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
        self.episode_history = []
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

            self.record_pi(states, pi, mask, mcts_value, network_value)

        return actions

    def get_actions_and_pi(self, action_distribution, states, tau):
        n_parallel_games = states.shape[0]
        pi = torch.zeros((n_parallel_games, self.game_class.get_n_actions()), device=states.device, dtype=torch.float32)
        actions = torch.zeros(n_parallel_games, dtype=torch.long, device=states.device)

        if type(tau) == float:
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

    def record_pi(self, states, pi, mask, mcts_value=None, network_value=None, env_state=False):
        # Record this for training, set the root node to the next node now and forget the parent
        self.episode_history.append(TimestepData(states, pi, mask, mcts_value, network_value))

    def on_rewards(self, rewards: torch.Tensor, mask: torch.Tensor):
        if self.training:
            self.episode_history.append(RewardData(rewards, mask))

    def before_env_move(self, states: torch.Tensor, mask: torch.Tensor):
        # Put this in the episode history because we're gonna learn the value for these states. pi will be nans
        num_actions = self.game_class.get_n_actions()
        self.record_pi(states,
                       torch.ones((states.shape[0], num_actions), dtype=torch.float32, device=states.device) * torch.nan,
                       mask,
                       env_state=True)

    def train(self):
        self.training = True
        self.hyperparams.n_mcts_iters.update_value('training', True)

    def eval(self):
        self.training = False
        self.hyperparams.n_mcts_iters.update_value('training', False)

    def before_game_start(self, game: GameMulti):
        self.game = game
        self.episode_history = []
        self.move_number_in_current_game = 0

    def on_game_end(self):
        if not self.training:
            return

        trajectories = self.get_trajectories()

        for data_listener in self.listeners:
            data_listener.on_trajectories(trajectories)

        if self.hyperparams.value_target_calculation_method == ValueTargetCalculationMethod.DISCOUNTED_REWARDS:
            self.generate_data_discounted_rewards_method()
        elif self.hyperparams.value_target_calculation_method == ValueTargetCalculationMethod.TD:
            self.generate_data_td_method_network_value()
        elif self.hyperparams.value_target_calculation_method == ValueTargetCalculationMethod.TD_MCTS:
            self.generate_data_td_method_mcts_value()
        elif self.hyperparams.value_target_calculation_method == ValueTargetCalculationMethod.TD_MCTS_NETWORK_FALLBACK:
            self.generate_data_td_method(lambda data: data.mcts_value if data.mcts_value is not None else data.network_value)

    def get_trajectories(self):
        state_trajectories = torch.zeros((0, self.game.n_parallel_games, *self.game_class.get_state_shape()),
                                         device=self.game.states.device, dtype=torch.float32)

        rewards = torch.zeros((0, self.game.n_parallel_games, self.game_class.get_n_players()), dtype=torch.float32,
                              device=self.game.states.device)

        search_values = torch.zeros((0, self.game.n_parallel_games, self.game_class.get_n_players()), dtype=torch.float32,
                                  device=self.game.states.device)

        pis = torch.zeros((0, self.game.n_parallel_games, self.game_class.get_n_actions()), dtype=torch.float32,
                          device=self.game.states.device)

        mask = torch.zeros((0, self.game.n_parallel_games), dtype=torch.bool, device=self.game.states.device)

        for data in self.episode_history:
            if isinstance(data, TimestepData):
                blank_states = torch.zeros((1, self.game.n_parallel_games, *self.game_class.get_state_shape()),
                                           device=self.game.states.device, dtype=torch.float32)
                state_trajectories = torch.cat((state_trajectories, blank_states))
                state_trajectories[-1, data.mask] = data.states

                pis = torch.cat((pis, torch.zeros((1, self.game.n_parallel_games, self.game_class.get_n_actions()),
                                                  device=self.game.states.device, dtype=torch.float32)))
                pis[-1, data.mask] = data.pis

                rewards = torch.cat((rewards,
                                     torch.zeros((1, self.game.n_parallel_games, self.game_class.get_n_players()),
                                                 device=self.game.states.device, dtype=torch.float32)))

                search_values = torch.cat((search_values,
                                         torch.zeros((1, self.game.n_parallel_games, self.game_class.get_n_players()),
                                                     device=self.game.states.device, dtype=torch.float32)))

                if data.mcts_value is not None:
                    search_values[-1, data.mask] = data.mcts_value
                else:
                    search_values[-1, data.mask] = torch.nan

                mask = torch.cat((mask, data.mask.unsqueeze(0)))

            elif isinstance(data, RewardData):
                rewards[-1, data.mask] = data.reward_value

        state_trajectories = state_trajectories.transpose(0, 1)
        rewards = rewards.transpose(0, 1)
        mask = mask.transpose(0, 1)
        pis = pis.transpose(0, 1)
        search_values = search_values.transpose(0, 1)

        trajectories = []

        for i in range(state_trajectories.shape[0]):
            trajectories.append(Trajectory(state_trajectories[i][mask[i]], rewards[i][mask[i]], pis[i][mask[i]],
                                           search_values[i][mask[i]]))

        return trajectories

    def generate_data_discounted_rewards_method(self):
        episode_history = reversed(self.episode_history)  # work backwards
        cum_rewards = torch.zeros((self.game.n_parallel_games, self.game_class.get_n_players()), dtype=torch.float32,
                                  device=self.game.states.device)

        for data in episode_history:
            if isinstance(data, RewardData):
                full_rewards = torch.zeros((self.game.n_parallel_games, self.game_class.get_n_players()),
                                           dtype=torch.float32, device=data.reward_value.device)
                full_rewards[data.mask] = data.reward_value
                cum_rewards = (full_rewards + self.hyperparams.discount * cum_rewards)

            elif isinstance(data, TimestepData):
                mask = data.mask
                reward = cum_rewards[torch.arange(self.game.n_parallel_games, device=mask.device)[mask], :]

                for data_listener in self.listeners:
                    data_listener.on_data_point(data.states, data.pis, reward)

    def generate_data_td_method_network_value(self):
        return self.generate_data_td_method(lambda data: data.network_value)

    def generate_data_td_method_mcts_value(self):
        return self.generate_data_td_method(lambda data: data.mcts_value)

    def generate_data_td_method(self, value_fn):
        """

        :param value_fn: A function that takes a TimestepData and returns the value to use for training. If the value is
        None, then the value is skipped in the TD summation

        """
        episode_history = reversed(self.episode_history)  # work backwards

        discounted_rewards_since_last_state = torch.zeros((self.game.n_parallel_games, self.game_class.get_n_players()),
                                                            dtype=torch.float32, device=self.game.states.device)

        last_state_vals = torch.zeros((self.game.n_parallel_games, self.game_class.get_n_players()),
                                      dtype=torch.float32, device=self.game.states.device)

        last_td_est = torch.zeros((self.game.n_parallel_games, self.game_class.get_n_players()),
                                  dtype=torch.float32, device=self.game.states.device)

        for data in episode_history:
            if isinstance(data, RewardData):
                full_rewards = torch.zeros((self.game.n_parallel_games, self.game_class.get_n_players()),
                                           dtype=torch.float32, device=data.reward_value.device)
                full_rewards[data.mask] = data.reward_value

                discounted_rewards_since_last_state = (full_rewards + self.hyperparams.discount *
                                                       discounted_rewards_since_last_state)
            elif isinstance(data, TimestepData):
                mask = data.mask

                if value_fn(data) is not None:
                    td_est = (discounted_rewards_since_last_state[mask]
                              + (1 - self.hyperparams.td_lambda.get_lambda()) * self.hyperparams.discount * last_state_vals[mask]
                              + self.hyperparams.td_lambda.get_lambda() * self.hyperparams.discount * last_td_est[mask])
                else:
                    td_est = last_td_est[mask]

                for data_listener in self.listeners:
                    data_listener.on_data_point(data.states, data.pis, td_est)

                if value_fn(data) is not None:
                    last_state_vals[mask] = value_fn(data)
                    discounted_rewards_since_last_state[mask] = 0
                    last_td_est[mask] = td_est


class TimestepData:
    def __init__(self, states, pis, mask, mcts_value=None, network_value=None, num_moves=None):
        self.states = states
        self.pis = pis
        self.mask = mask
        self.mcts_value = mcts_value
        self.network_value = network_value
        self.num_moves = num_moves

    def __repr__(self):
        return (f'(state: {self.states}, pi={self.pis}, mask={self.mask}'
                f'{f", mcts_val={self.mcts_value}" if self.mcts_value is not None else ""}'
                f'{f", net_val={self.network_value}" if self.network_value is not None else ""})'
                f'{f", num_moves={self.num_moves}" if self.num_moves is not None else ""}')


class RewardData:
    def __init__(self, reward_value, mask):
        self.reward_value = reward_value
        self.mask = mask

    def __repr__(self):
        return f'(r: {self.reward_value}, mask={self.mask})'


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
