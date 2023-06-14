from ..game import SequentialGame, GameListener
from typing import Type, List
from .training_manager import TrainingListener, ListDataset
from ..agent.alpha_agent import BaseAlphaEvaluator, AlphaAgent, AlphaAgentListener, AlphaAgentHyperparameters
from typing import Callable
import torch.optim
import torch.utils.data
from tqdm.auto import tqdm
import copy
import torch.nn as nn
import multiprocessing as mp
import queue


class AlphaDataset(AlphaAgentListener):
    def sample_minibatch(self, batch_size):
        raise NotImplementedError()


class BasicAlphaDataset(AlphaDataset):
    def __init__(self, evaluator: BaseAlphaEvaluator=None, max_size=50000, process_state=True, min_size=100):
        self.evaluator = evaluator
        self.max_size = max_size
        self.min_size = min_size
        self.states = []
        self.pis = []
        self.rewards = []
        self.process_state = process_state

        if process_state and (evaluator is None):
            raise ValueError('If process_state==True, you must give an evaluator.')

    def on_data_point(self, state, pi, reward):
        if self.process_state:
            state = self.evaluator.process_state(state)

        self.states.append(torch.FloatTensor(state))
        self.pis.append(torch.FloatTensor(pi))
        self.rewards.append(torch.FloatTensor([reward]))

        while len(self) > self.max_size:
            self.pop()

    def __len__(self):
        return len(self.states)

    def pop(self):
        self.states.pop()
        self.pis.pop()
        self.rewards.pop()

    def sample_minibatch(self, batch_size):
        dataset = ListDataset(self.states, self.pis, self.rewards)
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=batch_size)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return next(iter(dataloader))


class AlphaDataSender(AlphaAgentListener):
    """
    Used to send data between processes
    """
    def __init__(self, queue: mp.Queue, evaluator):
        self.queue = queue
        self.evaluator = evaluator

    def on_data_point(self, state, pi, reward):
        self.queue.put((self.evaluator.process_state(state), pi, reward))


class AlphaDataRelayer:
    def __init__(self, in_queue: mp.Queue, out_queue: mp.Queue, dataset: BasicAlphaDataset, minibatch_size: int):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.dataset = dataset
        self.minibatch_size = minibatch_size

    def check_queue_and_callback(self):
        try:
            state, pi, reward = self.in_queue.get_nowait()
        except queue.Empty:
            return
        else:
            self.dataset.on_data_point(state, pi, reward)

    def sample_minibatch_and_add_to_queue(self):
        if len(self.dataset) < self.dataset.min_size:
            return

        minibatch = self.dataset.sample_minibatch(self.minibatch_size)

        try:
            self.out_queue.put_nowait(minibatch)
        except queue.Full:
            pass


class AlphaNetworkEvaluator(BaseAlphaEvaluator):
    def __init__(self, network: nn.Module):
        self.network = network

    def evaluate(self, state):
        with torch.no_grad():
            self.network.eval()  # just be aware
            pi, v = self.network(self.process_state(state).unsqueeze(0))
        return pi.numpy(), v.numpy()

    def process_state(self, state):
        raise NotImplementedError()

    def eval(self):
        # Switch to eval mode
        self.network.eval()

    def train(self):
        # Switch to train mode
        self.network.train()


class AlphaNetworkOptimizerMonitor:
    def on_optimizer_step(self, loss):
        raise NotImplementedError()


class AlphaNetworkOptimizer:
    def __init__(self, evaluator: AlphaNetworkEvaluator, optimizer_constructor, monitors: List[AlphaNetworkOptimizerMonitor] = None):
        self.evaluator = evaluator
        self.optimizer = optimizer_constructor(self.evaluator.network.parameters())
        self.monitors = monitors

    def loss(self, processed_states, action_distns, values):
        pred_distns, pred_values = self.evaluator.network(processed_states)

        if pred_distns.shape != action_distns.shape or values.shape != pred_values.shape:
            raise ValueError('Shape mismatch between data and network predictions')

        return (values - pred_values) ** 2 - torch.sum(action_distns * torch.log(pred_distns), dim=1)

    def take_training_step_processed(self, processed_states, action_distns, values):
        # Run through network and compute loss
        losses = self.loss(processed_states, action_distns, values)
        loss = torch.sum(losses)

        # Backprop
        self.evaluator.network.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_item = losses.mean().item()

        if self.monitors is not None:
            for monitor in self.monitors:
                monitor.on_optimizer_step(loss_item)

        return loss_item


class AlphaTrainingHyperparameters(AlphaAgentHyperparameters):
    __slots__ = ['min_data_size', 'max_data_size', 'batch_size', 'share_among_players', 'lr', 'weight_decay',
                 'num_samples_per_epoch']

    def __init__(self):
        super().__init__()
        self.min_data_size = 1000
        self.max_data_size = 50000
        self.batch_size = 32
        self.share_among_players = True
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.num_samples_per_epoch = 100  # this is used by the dataset


class AlphaTrainingRun(GameListener):
    def __init__(self, game: Type[SequentialGame], alpha_evaluator: AlphaNetworkEvaluator,
                 optimizer_constructor: Callable[[], torch.optim.Optimizer],
                 hyperparams: AlphaTrainingHyperparameters,
                 training_listeners: List[TrainingListener] = ()):
        """

        :param game:
        :param network: Pass in a callable that returns a network
        :param optimizer: Pass in a function that will do f( network ) -> optimizer
        :param training_listeners:
        """
        self.game = game
        self.alpha_evaluators = []
        self.optimizers = []
        self.datasets = []
        self.listeners = training_listeners
        self.hyperparams = hyperparams
        self.agents = []
        self.n_iters = 0
        self.training_listeners = training_listeners

        for i in range(self.game.get_n_players()):
            if i == 0 or (not hyperparams.share_among_players):
                self.alpha_evaluators.append(copy.deepcopy(alpha_evaluator))
                self.optimizers.append( AlphaNetworkOptimizer(self.alpha_evaluators[-1], optimizer_constructor) )
                self.datasets.append( BasicAlphaDataset( self.alpha_evaluators[-1], max_size=self.hyperparams.max_data_size) )

            cur_agent = AlphaAgent(self.game, self.alpha_evaluators[-1], self.hyperparams, listeners=[self.datasets[-1]])
            self.agents.append(cur_agent)

        self.n_iters = 0
        for listener in self.listeners:
            listener.before_begin_training(self)

    def train(self, n_games=10000):
        game = self.game(self.agents, [self, *self.listeners])
        for _ in tqdm(range(n_games)):
            game.play()

    def after_action(self, game):
        for i, optimizer in enumerate(self.optimizers):
            if len(self.datasets[i]) > self.hyperparams.min_data_size:
                minibatch = self.datasets[i].sample_minibatch(self.hyperparams.batch_size)
                loss = optimizer.take_training_step_processed(*minibatch)

                self.n_iters += 1

                for training_listener in self.training_listeners:
                    training_listener.on_training_step(self.n_iters, loss, self)