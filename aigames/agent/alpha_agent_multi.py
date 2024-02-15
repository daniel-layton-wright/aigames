from dataclasses import dataclass
from .agent import AgentMulti
from typing import Type, List
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


class AlphaAgentMultiListener:
    def after_mcts_search(self, root_node):
        pass

    def on_data_point(self, state, pi, reward):
        pass


class AlphaAgentDebugListener(AlphaAgentMultiListener):
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


@dataclass(kw_only=True, slots=True)
class AlphaAgentHyperparametersMulti(MCTSHyperparameters):
    training_tau: TrainingTau = TrainingTau(1.0)
    use_dirichlet_noise_in_eval: bool = False  # the AlphaGo paper is unclear about this
    reuse_mcts_tree: bool = False  # Initial testing showed this was actually slower (on G2048Multi) :/


# noinspection PyBroadException
class AlphaAgentMulti(AgentMulti):
    """
    This version of the alpha agent works with multiple games being played at once (so we can use the faster
    version of MCTS). It's root parallelization.
    """
    def __init__(self, game_class: Type[GameMulti], evaluator: BaseAlphaEvaluator,
                 hyperparams: AlphaAgentHyperparametersMulti, listeners: List[AlphaAgentMultiListener] = None,
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

    def get_actions(self, states, mask):
        n_parallel_games = states.shape[0]

        if self.hyperparams.reuse_mcts_tree and self.mcts is not None:
            self.mcts = self.mcts.get_next_mcts(states)
        else:
            self.mcts = MCTS(self.game, self.evaluator, self.hyperparams, states)

        # Add Dirichlet noise to the root node
        if self.training or self.hyperparams.use_dirichlet_noise_in_eval:
            self.mcts.add_dirichlet_noise()

        self.mcts.search_for_n_iters(self.hyperparams.n_iters)

        if self.training:
            tau = self.hyperparams.training_tau.get_tau(self.move_number_in_current_game)
        else:
            tau = 0.0

        if self.hyperparams.n_iters > 1:
            action_distribution = self.mcts.n[:, 1]
        else:
            self.mcts.expand()
            action_distribution = self.mcts.pi[:, 1]

        pi = torch.zeros((n_parallel_games, self.game_class.get_n_actions()))

        if tau > 0:
            pi = action_distribution ** (1. / tau)
            pi /= pi.sum(dim=1, keepdim=True)
            # Choose the action according to the distribution pi
            actions = torch.multinomial(pi, num_samples=1).flatten()
        else:
            pi[torch.arange(n_parallel_games), action_distribution.argmax(dim=1)] = 1
            actions = action_distribution.argmax(dim=1).flatten()

        self.record_pi(mask, pi, states)

        return actions

    def record_pi(self, mask, pi, states):
        # Record this for training, set the root node to the next node now and forget the parent
        full_pi = torch.ones((mask.shape[0], *pi.shape[1:]), dtype=pi.dtype, device=pi.device) * torch.nan
        full_states = torch.ones((mask.shape[0], *states.shape[1:]), dtype=states.dtype,
                                 device=states.device) * torch.nan
        player_index = torch.ones((mask.shape[0],), dtype=torch.long, device=states.device) * -1
        full_states[mask] = states
        player_index[mask] = self.game.get_cur_player_index(states)
        full_pi[mask] = pi
        self.episode_history.append(TimestepData(full_states, full_pi, player_index))

    def on_rewards(self, rewards: torch.Tensor, mask: torch.Tensor):
        full_rewards = (torch.ones((mask.shape[0], *rewards.shape[1:]), dtype=rewards.dtype, device=rewards.device)
                        * torch.nan)
        full_rewards[mask] = rewards
        self.episode_history.append(RewardData(full_rewards))

    def train(self):
        self.evaluator.train()
        self.training = True

    def eval(self):
        self.evaluator.eval()
        self.training = False

    def before_game_start(self, game: GameMulti):
        self.game = game
        self.episode_history = []
        self.move_number_in_current_game = 0

    def on_game_end(self):
        if not self.training:
            return

        episode_history = reversed(self.episode_history)  # work backwards
        cum_rewards = torch.zeros((self.game.n_parallel_games, self.game_class.get_n_players()), dtype=torch.float32,
                                  device=self.game.states.device)

        for data in episode_history:
            if isinstance(data, RewardData):
                cum_rewards = (data.reward_value.nan_to_num(0) + self.hyperparams.discount * cum_rewards)
            elif isinstance(data, TimestepData):
                mask = data.player_index >= 0
                reward = cum_rewards[torch.arange(self.game.n_parallel_games, device=mask.device)[mask], :]
                for data_listener in self.listeners:
                    data_listener.on_data_point(data.states[mask], data.pis[mask], reward)


class TimestepData:
    def __init__(self, states, pis, player_index):
        self.states = states
        self.pis = pis
        self.player_index = player_index

    def __repr__(self):
        return f'(state: {self.states}, pi={self.pis}, i={self.player_index})'


class RewardData:
    def __init__(self, reward_value):
        self.reward_value = reward_value

    def __repr__(self):
        return f'(r: {self.reward_value})'


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
