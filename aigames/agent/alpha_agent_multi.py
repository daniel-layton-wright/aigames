import enum
from dataclasses import dataclass
from .agent import AgentMulti
from typing import Type, List
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


class AlphaAgentMultiListener:
    def after_mcts_search(self, mcts, pi, chosen_actions):
        pass

    def on_data_point(self, state, pi, reward):
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

        self.self_play_round_number = 0

    def get_tau(self, move_number):
        if self.fixed_tau_value is not None:
            return self.fixed_tau_value
        elif self.tau_schedule_list is not None:
            return self.tau_schedule_list[min(len(self.tau_schedule_list) - 1, move_number)]
        else:
            return self.tau_schedule_function(move_number, self.self_play_round_number)

    def update_self_play_round(self, i):
        self.self_play_round_number = i

    def __json__(self):
        return self.__dict__


class ValueTargetCalculationMethod(enum.Enum):
    DISCOUNTED_REWARDS = 'discounted_rewards'
    TD = 'td'
    TD_MCTS = 'td_mcts'
    TD_MCTS_NETWORK_FALLBACK = 'td_mcts_network_fallback'

    def __json__(self):
        return self.value


@dataclass(kw_only=True, slots=True)
class AlphaAgentHyperparametersMulti(MCTSHyperparameters):
    training_tau: TrainingTau = TrainingTau(1.0)
    use_dirichlet_noise_in_eval: bool = False  # the AlphaGo paper is unclear about this -> defintely should be False just think about it, don't need no stinking paper
    reuse_mcts_tree: bool = False  # Initial testing showed this was actually slower (on G2048Multi) :/
    value_target_calculation_method: ValueTargetCalculationMethod = ValueTargetCalculationMethod.TD
    td_lambda: float = 0.5  # only used if value_target_calculation_method is TD. Setting this to 1 is equivalent to discounted rewards
    full_mcts_probability: float = 1.0  # The probability of doing a full MCTS search, otherwise use n_fast_mcts_iters
    n_fast_mcts_iters: int = 20  # The number of MCTS iters to do when not doing a full MCTS search


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

        if self.hyperparams.full_mcts_probability < 1.0:
            # Create a copy of the hyperparams with the fast_n_mcts_iters to use when doing fast mode
            fast_mcts_args = {k: getattr(self.hyperparams, k) for k in MCTSHyperparameters.__slots__}
            fast_mcts_args['n_mcts_iters'] = self.hyperparams.n_fast_mcts_iters
            self.fast_mcts_hyperparams = MCTSHyperparameters(**fast_mcts_args)

    def get_actions(self, states, mask):
        n_parallel_games = states.shape[0]
        doing_fast_mcts = False

        if (self.training  # Only do fast mcts in training mode; in eval mode, do the full mcts at every move
                and self.hyperparams.full_mcts_probability < 1.0
                and torch.rand(1).item() > self.hyperparams.full_mcts_probability):
            mcts_hypers = self.fast_mcts_hyperparams
            doing_fast_mcts = True
        else:
            mcts_hypers = self.hyperparams

        if self.hyperparams.reuse_mcts_tree and self.mcts is not None:
            self.mcts = self.mcts.get_next_mcts(states)
        else:
            self.mcts = MCTS(self.game, self.evaluator, mcts_hypers, states,
                             add_dirichlet_noise=(self.training or self.hyperparams.use_dirichlet_noise_in_eval))

        self.mcts.search_for_n_iters(mcts_hypers.n_mcts_iters)

        if self.training:
            tau = self.hyperparams.training_tau.get_tau(self.move_number_in_current_game)
        else:
            tau = 0.0

        if self.hyperparams.n_mcts_iters > 1:
            action_distribution = self.mcts.n[:, 1]
        else:
            self.mcts.expand()
            action_distribution = self.mcts.pi[:, 1]

        pi = torch.zeros((n_parallel_games, self.game_class.get_n_actions()), device=states.device, dtype=torch.float32)

        if tau > 0:
            pi = action_distribution ** (1. / tau)
            pi /= pi.sum(dim=1, keepdim=True)
            # Choose the action according to the distribution pi
            actions = torch.multinomial(pi, num_samples=1).flatten()
        else:
            pi[torch.arange(n_parallel_games), action_distribution.argmax(dim=1)] = 1
            actions = action_distribution.argmax(dim=1).flatten()

        mcts_value = self.mcts.w[:, 1].sum(dim=1) / self.mcts.n[:, 1].sum(dim=1).unsqueeze(-1)
        network_value = self.mcts.values[:, 1]

        for listener in self.listeners:
            listener.after_mcts_search(self.mcts, pi, actions)

        if self.training:
            if doing_fast_mcts:
                # If doing fast MCTS, do not use the pi or mcts_value to train
                pi *= torch.nan
                mcts_value = None

            self.record_pi(states, pi, mask, mcts_value, network_value)

        return actions

    def record_pi(self, states, pi, mask, mcts_value=None, network_value=None):
        # Record this for training, set the root node to the next node now and forget the parent
        self.episode_history.append(TimestepData(states, pi, mask, mcts_value, network_value))

    def on_rewards(self, rewards: torch.Tensor, mask: torch.Tensor):
        self.episode_history.append(RewardData(rewards, mask))

    def before_env_move(self, states: torch.Tensor, mask: torch.Tensor):
        # Put this in the episode history because we're gonna learn the value for these states. pi will be nans
        num_actions = self.game_class.get_n_actions()
        self.record_pi(states,
                       torch.ones((states.shape[0], num_actions), dtype=torch.float32, device=states.device) * torch.nan,
                       mask)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def before_game_start(self, game: GameMulti):
        self.game = game
        self.episode_history = []
        self.move_number_in_current_game = 0

    def on_game_end(self):
        if not self.training:
            return

        if self.hyperparams.value_target_calculation_method == ValueTargetCalculationMethod.DISCOUNTED_REWARDS:
            self.generate_data_discounted_rewards_method()
        elif self.hyperparams.value_target_calculation_method == ValueTargetCalculationMethod.TD:
            self.generate_data_td_method_network_value()
        elif self.hyperparams.value_target_calculation_method == ValueTargetCalculationMethod.TD_MCTS:
            self.generate_data_td_method_mcts_value()
        elif self.hyperparams.value_target_calculation_method == ValueTargetCalculationMethod.TD_MCTS_NETWORK_FALLBACK:
            self.generate_data_td_method(lambda data: data.mcts_value if data.mcts_value is not None else data.network_value)

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
                              + (1 - self.hyperparams.td_lambda) * self.hyperparams.discount * last_state_vals[mask]
                              + self.hyperparams.td_lambda * self.hyperparams.discount * last_td_est[mask])
                else:
                    td_est = last_td_est[mask]

                for data_listener in self.listeners:
                    data_listener.on_data_point(data.states, data.pis, td_est)

                if value_fn(data) is not None:
                    last_state_vals[mask] = value_fn(data)
                    discounted_rewards_since_last_state[mask] = 0
                    last_td_est[mask] = td_est


class TimestepData:
    def __init__(self, states, pis, mask, mcts_value=None, network_value=None):
        self.states = states
        self.pis = pis
        self.mask = mask
        self.mcts_value = mcts_value
        self.network_value = network_value

    def __repr__(self):
        return (f'(state: {self.states}, pi={self.pis}, mask={self.mask}'
                f'{f", mcts_val={self.mcts_value}" if self.mcts_value is not None else ""}'
                f'{f", net_val={self.network_value}" if self.network_value is not None else ""})')


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
