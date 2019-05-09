import sys
from aigames.agents.alpha_agent import *
import torch.multiprocessing as mp
from ctypes import c_bool


def self_play(game_class, agent, n_games: int):
    players = [agent for _ in range(game_class.N_PLAYERS)]
    for _ in range(n_games):
        cur_game = game_class(players)
        cur_game.play()


def evaluation_worker(model, device, evaluation_queue, results_queues, kill):
    worker = MultiprocessingAlphaEvaluationWorker(model, device, evaluation_queue, results_queues, kill)
    worker.evaluate_until_killed()


def training_worker(model, optimizer, device, train_queue, kill, monitor=None, pause_training=None):
    if monitor is not None:
        monitor.before_training_start()

    worker = MultiprocessingAlphaTrainingWorker(model, optimizer, device, train_queue, kill, monitor, pause_training)
    worker.train_until_killed()


def monitor_worker(monitor: MultiprocessingAlphaMonitor):
    monitor.start()
    monitor.monitor_until_killed()


class MultiprocessingManager:
    def __init__(self, game_class, model, model_device, optimizer, monitor,
                 n_self_play_workers: int, n_evaluation_workers: int, n_training_workers: int,
                 n_games_per_worker: int, evaluation_queue_max_size, train_queue_max_size,
                 alpha_agent_kwargs=dict()):
        self.game_class = game_class
        self.model = model
        self.model_device = model_device
        self.optimizer = optimizer
        self.monitor = monitor
        self.n_self_play_workers = n_self_play_workers
        self.n_evaluation_workers = n_evaluation_workers
        self.n_training_workers = n_training_workers
        self.n_games_per_worker = n_games_per_worker
        self.alpha_agent_kwargs = alpha_agent_kwargs

        self.evaluation_workers = []
        self.training_workers = []
        self.self_play_workers = []
        self.monitor_workers = []

        self.evaluation_queue = mp.Queue(maxsize=evaluation_queue_max_size)
        self.train_queue = mp.Queue(maxsize=train_queue_max_size)
        self.results_queues = [mp.Queue(maxsize=1) for _ in
                               range(self.n_self_play_workers + int(self.monitor is not None))]
        self.kill_monitor = mp.Value(c_bool, False)
        self.kill_eval_train = mp.Value(c_bool, False)
        self.pause_training = mp.Value(c_bool, False)

        if self.monitor is not None:
            monitor_evaluator = MultiprocessingAlphaEvaluator(len(self.results_queues) - 1, self.model,
                                                              self.model_device, self.evaluation_queue,
                                                              self.results_queues[-1], self.train_queue)
            monitor_agent = AlphaAgent(self.game_class, monitor_evaluator, **self.alpha_agent_kwargs)
            self.monitor = monitor(self.model, monitor_agent, self.kill_monitor, self.pause_training)

    def start(self):
        self.start_evaluation_workers()
        self.start_training_workers()
        self.start_self_play_workers()
        self.start_monitor_workers()

    def wait_for_self_play_workers(self):
        for worker in self.self_play_workers:
            worker.join()

    def terminate_all(self):
        for worker in self.evaluation_workers + self.training_workers + self.self_play_workers + self.monitor_workers:
            worker.terminate()

    def kill_auxiliary_workers(self):
        self.kill_monitor.value = True
        for worker in self.monitor_workers:
            worker.join()

        self.kill_eval_train.value = True
        for worker in self.evaluation_workers + self.training_workers:
            worker.join()

    def start_evaluation_workers(self):
        for i in range(self.n_evaluation_workers):
            cur_evaluation_worker = mp.Process(target=evaluation_worker,
                                                args=(
                                                    self.model, self.model_device, self.evaluation_queue,
                                                    self.results_queues, self.kill_eval_train))
            cur_evaluation_worker.start()
            self.evaluation_workers.append(cur_evaluation_worker)

    def start_training_workers(self):
        for i in range(self.n_training_workers):
            cur_training_worker = mp.Process(target=training_worker,
                                              args=(self.model, self.optimizer, self.model_device, self.train_queue,
                                                    self.kill_eval_train, self.monitor, self.pause_training))
            cur_training_worker.start()
            self.training_workers.append(cur_training_worker)

    def start_self_play_workers(self):
        for i in range(self.n_self_play_workers):
            cur_evaluator = MultiprocessingAlphaEvaluator(i, self.model, self.model_device, self.evaluation_queue,
                                                          self.results_queues[i], self.train_queue, self.pause_training)
            cur_agent = AlphaAgent(self.game_class, cur_evaluator, **self.alpha_agent_kwargs)
            cur_worker = mp.Process(target=self_play, args=(self.game_class, cur_agent, self.n_games_per_worker))
            cur_worker.start()
            self.self_play_workers.append(cur_worker)

    def start_monitor_workers(self):
        if self.monitor is not None:
            cur_monitor_worker = mp.Process(target=monitor_worker, args=(self.monitor,))
            cur_monitor_worker.start()
            self.monitor_workers.append(cur_monitor_worker)


def train_alpha_agent_mp(game_class, model: AlphaModel, optimizer_class, lr=0.01, monitor=None, model_device='cpu',
                         n_self_play_workers=1,
                         n_games_per_worker=10000,
                         n_evaluation_workers=1, n_training_workers=1,
                         evaluation_queue_max_size=100, train_queue_max_size=100,
                         alpha_agent_kwargs=dict(),
                         ):
    if type(model_device) == str:
        model_device = torch.device(model_device)
    model.share_memory()
    model.to(model_device)

    optimizer = optimizer_class(model.parameters(), lr=lr)
    mp_manager = MultiprocessingManager(game_class, model, model_device, optimizer, monitor, n_self_play_workers,
                                        n_evaluation_workers, n_training_workers, n_games_per_worker,
                                        evaluation_queue_max_size, train_queue_max_size, alpha_agent_kwargs)

    try:
        mp_manager.start()
        mp_manager.wait_for_self_play_workers()
        mp_manager.kill_auxiliary_workers()
    except KeyboardInterrupt:
        mp_manager.terminate_all()
