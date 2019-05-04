from aigames.agents.alpha_agent import *
from aigames import AlphaMonitor
import torch.multiprocessing as mp


def self_play(game_class, agent, n_games: int):
    for _ in range(n_games):
        cur_game = game_class([agent, agent])
        cur_game.play()


def evaluation_worker(model, device, evaluation_queue, results_queues, kill_queue):
    worker = MultiprocessingAlphaEvaluationWorker(model, device, evaluation_queue, results_queues, kill_queue)
    worker.evaluate_until_killed()


def training_worker(model, optimizer, device, train_queue, kill_queue, monitor=None):
    if monitor is not None:
        monitor.before_training_start()

    worker = MultiprocessingAlphaTrainingWorker(model, optimizer, device, train_queue, kill_queue, monitor)
    worker.train_until_killed()


def train_alpha_agent_mp(game_class, model, monitor=None, model_device='cpu', n_self_play_procs=1,
                         n_games_per_proc=10000,
                         n_evaluation_workers=1, n_training_workers=1, evaluation_queue_max_size=100,
                         train_queue_max_size=100,
                         alpha_agent_kwargs=dict(),
                         ):
    evaluation_queue = mp.Queue(maxsize=evaluation_queue_max_size)
    train_queue = mp.Queue(maxsize=train_queue_max_size)
    results_queues = [mp.Queue(maxsize=1) for _ in range(n_self_play_procs)]
    kill_queue = mp.Queue(maxsize=1)

    optimizer = torch.optim.Adam(model.parameters())

    evaluation_processes = []
    training_processes = []
    self_play_processes = []

    try:
        for i in range(n_evaluation_workers):
            cur_evaluation_process = mp.Process(target=evaluation_worker,
                                                args=(model, model_device, evaluation_queue, results_queues, kill_queue))
            cur_evaluation_process.start()
            evaluation_processes.append(cur_evaluation_process)

        for i in range(n_training_workers):
            cur_training_process = mp.Process(target=training_worker,
                                              args=(model, optimizer, model_device, train_queue, kill_queue, monitor))
            cur_training_process.start()
            training_processes.append(cur_training_process)

        for i in range(n_self_play_procs):
            cur_evaluator = MultiprocessingAlphaEvaluator(i, model, evaluation_queue, results_queues[i], train_queue)
            cur_agent = AlphaAgent(game_class, cur_evaluator, **alpha_agent_kwargs)
            cur_process = mp.Process(target=self_play, args=(game_class, cur_agent, n_games_per_proc))
            cur_process.start()
            self_play_processes.append(cur_process)

        for sc in self_play_processes:
            sc.join()

        kill_queue.put(True)
        for sc in evaluation_processes:
            sc.join()
        for sc in training_processes:
            sc.join()

    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPT. SHUTTING DOWN...')
        for process in evaluation_processes + training_processes + self_play_processes:
            print(' TERMINATING PROCESS...')
            process.terminate()
