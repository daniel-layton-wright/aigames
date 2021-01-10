from ..agent.qlearning_agent import *
from ..game import *
import torch.nn as nn
import torch.multiprocessing as mp
import torch
from typing import List
from .qlearning_training_manager import take_training_step, BasicQLearningDataset, TrainingListener, DebugGameListener
import queue
from tqdm.auto import tqdm
from ctypes import c_bool, c_int
import signal
import sys


class MPQLearningTrainingRun(GameListener):
    def __init__(self, game: Type[PartiallyObservableSequentialGame], network, optimizer, state_shape,
                 n_self_play_workers: int = 2, n_training_workers: int = 1,
                 exploration_probability_half_life=1000, min_exploration_probability: float = 0.05,
                 starting_exploration_probability: float = 1,
                 discount_rate=0.99, update_target_network_ever_n_iters=1,
                 min_data_size=1000, max_data_size=50000, batch_size=32, share_among_players=True,
                 frac_terminal_to_sample=None, training_listeners: List[TrainingListener] = (),
                 sample_proportional_to_reward=False, device=torch.device('cpu'),
                 data_debug_mode=False, max_data_queue_size=100):
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
        self.share_among_players = share_among_players
        self.networks = []
        self.optimizers = []
        self.state_shape = state_shape
        self.datasets = []
        self.listeners = training_listeners
        self.discount_rate = discount_rate
        self.min_data_size = min_data_size
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.agents = []
        self.evaluation_queues = []
        self.result_queues = []
        self.data_queues = []
        self.minibatch_queues = []
        self.update_target_network_ever_n_iters = update_target_network_ever_n_iters
        self.n_iters = []
        self.n_self_play_workers = n_self_play_workers
        self.n_training_workers = n_training_workers
        self.device = device
        self.loss_queues = []
        self.frac_terminal_to_sample = frac_terminal_to_sample
        self.sample_proportional_to_reward = sample_proportional_to_reward
        self.stop_sentinel = mp.Value(c_bool, False)
        self.exploration_probability_half_life = exploration_probability_half_life
        self.exploration_probability_schedulers = []

        for i in range (self.game.get_n_players()):
            if i == 0 or not share_among_players:
                self.networks.append( network() )
                self.networks[-1].share_memory()
                self.optimizers.append( optimizer(self.networks[-1]) )
                self.datasets.append( BasicQLearningDataset(self.game, self.state_shape, max_size=max_data_size,
                                                            debug_mode=data_debug_mode))
                self.evaluation_queues.append( mp.Queue() )
                self.data_queues.append( mp.Queue(maxsize=max_data_queue_size) )
                self.minibatch_queues.append( mp.Queue(maxsize=n_training_workers) )
                self.result_queues.append( [] )
                self.n_iters.append( mp.Value(c_int, 0) )
                cur_exp_prob_scheduler = MPExponentialDecayExplorationProbabilityScheduler(
                                            exploration_probability_half_life,
                                            self.n_iters[-1], min_exploration_probability=min_exploration_probability,
                                            starting_exploration_probability=starting_exploration_probability)
                self.exploration_probability_schedulers.append( cur_exp_prob_scheduler )
                self.loss_queues.append( mp.Queue() )

        for i in range(n_self_play_workers * self.game.get_n_players()):
            network_num = 0 if share_among_players else (i % self.game.get_n_players())
            result_queue = mp.Queue()
            self.result_queues[network_num].append(result_queue)
            q_function = MPQLearningFunction(len(self.result_queues[network_num])-1, self.evaluation_queues[network_num],
                                             result_queue, self.stop_sentinel)
            data_listener = MPQLearningDataListener(self.data_queues[network_num])
            cur_agent = QLearningAgent(self.game, (i % self.game.get_n_players()), q_function,
                                       data_listener=data_listener,
                                       exploration_probability_scheduler=self.exploration_probability_schedulers[network_num],
                                       discount_rate=self.discount_rate)
            self.agents.append(cur_agent)

    def train(self, n_games=10000):
        for listener in self.listeners:
            listener.before_begin_training(self)

        n_games_left = mp.Value(c_int, n_games)
        self.stop_sentinel.value = False

        subprocs = []

        evaluation_worker = mp.Process(target=network_evaluator_worker, args=(self.networks, self.evaluation_queues,
                                                                              self.result_queues, self.stop_sentinel,
                                                                              self.state_shape,
                                                                              self.device))
        evaluation_worker.start()
        subprocs.append(('eval', evaluation_worker))

        sample_kwargs = {'batch_size': self.batch_size, 'frac_terminal': self.frac_terminal_to_sample,
                         'sample_proportional_to_reward': self.sample_proportional_to_reward}
        data_proc = mp.Process(target=data_worker, args=(self.datasets, sample_kwargs, self.min_data_size,
                                                         self.data_queues, self.minibatch_queues,
                                                         self.stop_sentinel))
        data_proc.start()
        subprocs.append(('data', data_proc))

        n_agents = self.game.get_n_players()

        for i in range(self.n_self_play_workers):
            game = self.game(self.agents[(i * n_agents): ((i + 1) * n_agents)], [])
            self_play_proc = mp.Process(target=self_play_worker, args=(game, n_games_left, self.stop_sentinel))
            self_play_proc.start()
            subprocs.append(('self_play', self_play_proc))

        for i in range(self.n_training_workers):
            training_proc = mp.Process(target=training_worker, args=(self.networks, copy.deepcopy(self.networks),
                                                                     self.optimizers, self.discount_rate,
                                                                     self.update_target_network_ever_n_iters,
                                                                     self.device, self.minibatch_queues,
                                                                     self.n_iters, self.stop_sentinel, self.loss_queues))
            training_proc.start()
            subprocs.append(('training', training_proc))

        def sigterm_handler(_signo, _stack_frame):
            sys.exit(0)

        signal.signal(signal.SIGTERM, sigterm_handler)
        try:
            while True:
                for i in range(len(self.networks)):
                    try:
                        iter, loss, terminal_loss, nonterminal_loss = self.loss_queues[i].get_nowait()
                        for listener in self.listeners:
                            listener.on_training_step(iter, loss, self, network_num=i, terminal_loss=terminal_loss,
                                                      nonterminal_loss=nonterminal_loss)
                    except queue.Empty:
                        pass

                if n_games_left.value <= 0:
                    break
        except KeyboardInterrupt:
            self.stop_sentinel.value = True
        finally:
            self.stop_sentinel.value = True

        for name, proc in subprocs:
            print(f'Waiting for {name}...')
            proc.join()

        print('Done.')

    def on_action(self, game, action):
        pass

    def play_debug_game(self):
        cli = CommandLineGame()
        debugger = DebugGameListener()

        for agent in self.agents:
            agent.eval()

        game = self.game(self.agents, listeners=[cli, debugger])
        game.play()


class MPQLearningFunction(QLearningFunction):
    def __init__(self, worker_num, evaluation_queue: mp.Queue, results_queue: mp.Queue, stop_sentinel: mp.Value):
        self.worker_num = worker_num
        self.evaluation_queue = evaluation_queue
        self.results_queue = results_queue
        self.stop_sentinel = stop_sentinel

    def evaluate(self, state):
        self.evaluation_queue.put((self.worker_num, state))
        while True:
            if self.stop_sentinel.value:
                raise AbortGameException()

            try:
                result = self.results_queue.get(timeout=0.1)
                break
            except queue.Empty:
                continue

        return result


class MPQLearningDataListener(QLearningDataListener):
    def __init__(self, data_queue: mp.Queue):
        self.data_queue = data_queue

    def on_SARS(self, state, action_index, reward, next_state):
        self.data_queue.put((state, action_index, reward, next_state))


class MPExponentialDecayExplorationProbabilityScheduler(ExplorationProbabilityScheduler):
    def __init__(self, half_life: float, i: mp.Value, min_exploration_probability: float = 0,
                 starting_exploration_probability: float = 1):
        self.half_life = half_life
        self.min_exploration_probability = min_exploration_probability
        self.starting_exploration_probability = starting_exploration_probability
        self.i = i

    def get_exploration_probability(self, agent, state):
        return (self.min_exploration_probability +
                (self.starting_exploration_probability - self.min_exploration_probability)
                * (0.5 ** (self.i.value / self.half_life)))


def network_evaluator_worker(networks: List[nn.Module], evaluation_queues: List[mp.Queue], results_queues: List[List[mp.Queue]],
                             stop_sentinel: mp.Value,
                             state_shape: tuple, device=torch.device('cpu')):
    while not stop_sentinel.value:
        assert(len(networks) == len(evaluation_queues) and len(networks) == len(results_queues))

        for i in range(len(networks)):
            try:
                worker_num, state = evaluation_queues[i].get_nowait()
            except queue.Empty:
                continue

            with torch.no_grad():
                result = networks[i](torch.Tensor(state).reshape(state_shape).unsqueeze(0).to(device)).squeeze()

            results_queues[i][worker_num].put(result)


def self_play_worker(game, n_games_left: mp.Value, stop_sentinel: mp.Value):
    while n_games_left.value > 0:
        if stop_sentinel.value:
            return
        game.play()
        n_games_left.value -= 1


def training_worker(networks: List[nn.Module], target_networks: List[nn.Module], optimizers: List,
                    discount_rate, update_target_network_ever_n_iters,
                    device, minibatch_queues: List[mp.Queue], iters: List[mp.Value], stop_sentinel: mp.Value,
                    loss_queues: List[mp.Queue]):
    last_updated_target_network = [0 for _ in range(len(networks))]
    while not stop_sentinel.value:
        for i in range(len(networks)):
            if iters[i].value - last_updated_target_network[i] > update_target_network_ever_n_iters:
                target_networks[i] = copy.deepcopy(networks[i])
                last_updated_target_network[i] = iters[i].value

            try:
                terminal_minibatch, nonterminal_minibatch = minibatch_queues[i].get_nowait()
            except queue.Empty:
                continue

            loss, terminal_loss, nonterminal_loss = take_training_step(terminal_minibatch, nonterminal_minibatch,
                                                                       target_networks[i], networks[i], optimizers[i],
                                                                       discount_rate, device)

            loss_queues[i].put((iters[i].value, loss, terminal_loss, nonterminal_loss))

            iters[i].value += 1


def data_worker(datasets: List[BasicQLearningDataset], sample_kwargs, min_data_size,
                incoming_data_queues: List[mp.Queue], outgoing_minibatch_queues: List[mp.Queue], stop_sentinel: mp.Value):
    while not stop_sentinel.value:
        for i in range(len(incoming_data_queues)):
            try:
                SARS = incoming_data_queues[i].get_nowait()
                datasets[i].on_SARS(*SARS)
            except queue.Empty:
                pass

            if not outgoing_minibatch_queues[i].full() and len(datasets[i]) >= min_data_size:
                minibatch = datasets[i].sample_minibatch(**sample_kwargs)
                try:
                    outgoing_minibatch_queues[i].put_nowait(minibatch)
                except queue.Full:
                    pass
