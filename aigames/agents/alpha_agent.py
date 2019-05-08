import torch
import torch.multiprocessing as mp
import torch.nn as nn
import queue
from aigames.base.game import *
from aigames.base.agent import *


class BaseAlphaEvaluator:
    def evaluate(self, state):
        raise NotImplementedError()

    def take_training_step(self, states, action_distns, values):
        raise NotImplementedError()

    def eval(self):
        # Switch to eval mode
        pass

    def train(self):
        # Switch to train mode
        pass


# TODO : add Dirichlet noise
class AlphaAgent(SequentialAgent):
    def __init__(self, game: Game, evaluator: BaseAlphaEvaluator,
                 training_tau: float = 1., c_puct: float = 1., n_mcts: int = 1200,
                 discount_rate: float = 1):
        super().__init__(game)
        self.game = game
        self.training_tau = training_tau
        self.tau = training_tau
        self.c_puct = c_puct
        self.n_mcts = n_mcts
        self.discount_rate = discount_rate
        self.evaluator = evaluator
        self.cur_node = None
        self.training = True
        self.episode_history = []

    def choose_action(self, state, player_index, verbose=False):
        if self.cur_node is None:
            self.cur_node = MCTSNode(self.game, state, None, self.evaluator, self.c_puct)
        elif (self.cur_node.state != state).any():
            self.cur_node = MCTSNode(self.game, state, None, self.evaluator, self.c_puct)

        for i in range(self.n_mcts):
            self.cur_node.search()

        pi = np.zeros(len(self.game.ALL_ACTIONS))
        if self.tau > 0:
            pi[self.cur_node.action_indices] = self.cur_node.children_total_N ** (1. / self.tau)
            pi /= np.sum(pi)
        else:
            pi[self.cur_node.action_indices[np.argmax(self.cur_node.children_total_N)]] = 1

        action_index = np.random.choice(range(len(self.game.ALL_ACTIONS)), p=pi)
        action = self.game.ALL_ACTIONS[action_index]
        child_index = self.cur_node.actions.index(action)

        self.episode_history.append(TimestepData(state, pi, player_index))
        self.cur_node = self.cur_node.children[child_index]
        self.cur_node.parent = None

        return action

    def reward(self, reward_value, state, player_index):
        self.episode_history.append(RewardData(player_index, reward_value))

    def train(self):
        self.tau = self.training_tau
        self.evaluator.train()
        self.training = True

    def eval(self):
        self.tau = 0
        self.evaluator.eval()
        self.training = False

    def start_episode(self):
        self.episode_history = []

    def end_episode(self):
        if not self.training:
            return

        episode_history = reversed(self.episode_history)
        cum_rewards = [0 for _ in range(self.game.N_PLAYERS)]
        states = []
        pis = []
        rewards = []

        for data in episode_history:
            if isinstance(data, RewardData):
                cum_rewards[data.player_index] = data.reward_value + self.discount_rate * cum_rewards[data.player_index]
            elif isinstance(data, TimestepData):
                reward = cum_rewards[data.player_index]

                states.append(data.state)
                pis.append(data.pi)
                rewards.append(torch.FloatTensor([reward]))

        self.evaluator.take_training_step(states, pis, rewards)


class MCTSNode:
    def __init__(self, game, state, parent_node, evaluator, c_puct, N=None, total_N=None, Q=None):
        self.game = game
        self.state = state
        self.parent = parent_node
        self.actions = self.game.legal_actions(self.state)
        self.next_states = [self.game.get_next_state(self.state, action) for action in self.actions]
        self.action_indices = self.game.legal_action_indices(self.state)
        self.n_children = len(self.actions)
        self.children = None
        self.children_N = np.zeros((self.n_children, self.game.N_PLAYERS))
        self.children_total_N = np.zeros(self.n_children)
        self.children_Q = np.zeros((self.n_children, self.game.N_PLAYERS))
        self.evaluator = evaluator
        self.c_puct = c_puct

        self.player_index = self.game.get_player_index(self.state)
        self.N = N if N is not None else np.zeros(self.game.N_PLAYERS)
        self.total_N = total_N if total_N is not None else np.array([0])
        self.Q = Q if Q is not None else np.zeros(self.game.N_PLAYERS)

        self.rewards = self.game.all_rewards(self.state)
        self.W = np.zeros(self.game.N_PLAYERS)
        self.P = None
        self.P_normalized = None
        self.v = None  # v is from the perspective of the player whose turn it is

    def search(self):
        if self.children is None:
            # We've reached a leaf node, expand
            self.expand()

            # backup
            if not self.game.is_terminal_state(self.state):
                self.backup(self.v, self.player_index)
            else:
                self.backup_all(self.rewards)
        else:
            # choose the node according to max Q + U
            U_values = self.c_puct * self.P_normalized * np.sqrt(self.total_N) / (1 + self.children_total_N)
            Q_plus_U_values = self.children_Q[:, self.player_index] + U_values
            next_node = self.children[np.argmax(Q_plus_U_values)]
            next_node.search()

    def expand(self):
        if not self.game.is_terminal_state(self.state):
            # Evaluate the node with the evaluator
            self.P, self.v = self.evaluator.evaluate(self.state)

            # Initialize the children
            self.P_normalized = self.P[self.action_indices].detach().cpu().numpy()
            self.P_normalized /= self.P_normalized.sum()

            self.children = [MCTSNode(self.game, next_state, self, self.evaluator, self.c_puct,
                                      self.children_N[i], self.children_total_N[i:(i + 1)], self.children_Q[i])
                             for i, next_state in enumerate(self.next_states)]

    def backup(self, value, player_index):
        self.N[player_index] += 1
        self.total_N += 1
        self.W[player_index] += value
        self.Q[player_index] = self.rewards[player_index] + self.W[player_index] / self.N[player_index]

        if self.parent is not None:
            self.parent.backup(value, player_index + self.rewards[player_index])

    def backup_all(self, values):
        self.N += 1
        self.total_N += 1
        self.W += values
        self.Q[:] = self.rewards + self.W / self.N
        if self.parent is not None:
            self.parent.backup_all(values)


class TimestepData:
    def __init__(self, state, pi, player_index):
        self.state = torch.FloatTensor(state)
        self.pi = torch.FloatTensor(pi)
        self.player_index = player_index


class RewardData:
    def __init__(self, player_index, reward_value):
        self.player_index = player_index
        self.reward_value = reward_value


class AlphaModel(nn.Module):
    @staticmethod
    def process_state(state):
        raise NotImplementedError()

    def loss(self, processed_states, action_distns, values):
        pred_distns, pred_values = self(processed_states)
        return (values - pred_values) ** 2 - torch.sum(action_distns * torch.log(pred_distns), dim=1)


class AlphaMonitor:
    def before_training_start(self):
        pass

    def on_game_end(self, alpha_agent: AlphaAgent):
        pass

    def on_optimizer_step(self, most_recent_loss):
        pass


class MultiprocessingAlphaMonitor(AlphaMonitor):
    def start(self):
        pass

    def monitor_until_killed(self):
        pass


class AlphaEvaluator(BaseAlphaEvaluator):
    def __init__(self, model, model_device, optimizer, monitor=None):
        self.model = model
        self.model_device = model_device
        self.optimizer = optimizer
        self.monitor = monitor

    def evaluate(self, state):
        processed_state = self.model.process_state(state).to(self.model_device)
        return self.model(processed_state)

    def take_training_step(self, states, action_distns, values):
        # Process states and move tensors to correct device
        processed_states = tuple(self.model.process_state(state) for state in states)
        processed_states = torch.cat(processed_states).to(self.model_device)
        action_distns = torch.stack(action_distns).to(self.model_device)
        values = torch.cat(tuple(values)).to(self.model_device)

        # Run through network and compute loss
        losses = self.model.loss(processed_states, action_distns, values)
        loss = torch.sum(losses)

        # Backprop
        loss.backward()
        self.optimizer.step()
        if self.monitor is not None:
            self.monitor.on_optimizer_step(losses.mean().item())

class MultiprocessingAlphaEvaluator(BaseAlphaEvaluator):
    def __init__(self, id, model: AlphaModel, model_device, evaluation_queue: mp.Queue, results_queue: mp.Queue,
                 train_queue: mp.Queue):
        self.id = id
        self.model = model
        self.model_device = model_device
        self.evaluation_queue = evaluation_queue
        self.results_queue = results_queue
        self.train_queue = train_queue

    def evaluate(self, state):
        processed_state = self.model.process_state(state).to(self.model_device)
        self.evaluation_queue.put((self.id, processed_state))
        pi, v = self.results_queue.get()
        pi_clone = pi.clone()
        v_clone = v.clone()
        del pi
        del v
        return pi_clone, v_clone

    def take_training_step(self, states, action_distns, values):
        processed_states = tuple(self.model.process_state(state) for state in states)
        processed_states = torch.cat(processed_states)
        action_distns = torch.stack(tuple(action_distns))
        values = torch.cat(tuple(values))
        self.train_queue.put(
            (processed_states.to(self.model_device),
             action_distns.to(self.model_device),
             values.to(self.model_device))
        )

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class MultiprocessingAlphaEvaluationWorker:
    def __init__(self, model: AlphaModel, device: torch.device, evaluation_queue: mp.Queue,
                 results_queues: List[mp.Queue],
                 kill: mp.Value):
        self.model = model
        self.device = device
        self.evaluation_queue = evaluation_queue
        self.results_queues = results_queues
        self.kill = kill

    def evaluate_until_killed(self):
        while True:
            if self.kill.value and self.evaluation_queue.empty():
                break

            try:
                id, processed_state = self.evaluation_queue.get(block=False)
            except queue.Empty:
                continue

            with torch.no_grad():
                result = self.model(processed_state.to(self.device))
                self.results_queues[id].put(result)


class MultiprocessingAlphaTrainingWorker:
    def __init__(self, model: AlphaModel, optimizer: torch.optim.Optimizer, device: torch.device, train_queue: mp.Queue,
                 kill: mp.Value, monitor: AlphaMonitor = None, pause_training=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_queue = train_queue
        self.kill = kill
        self.monitor = monitor
        self.pause_training = pause_training

    def train_until_killed(self):
        while True:
            try:
                processed_states, action_distns, values = self.train_queue.get(block=False)
            except queue.Empty:
                if self.kill.value:
                    break
                else:
                    continue

            while self.pause_training is not None and self.pause_training.value:
                pass

            # Run through network and compute loss
            losses = self.model.loss(processed_states.to(self.device), action_distns.to(self.device),
                                     values.to(self.device))
            loss = torch.sum(losses)

            # Backprop
            loss.backward()
            self.optimizer.step()

            if self.monitor is not None:
                self.monitor.on_optimizer_step(losses.mean().item())
