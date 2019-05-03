import os
import sys
top_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))
sys.path.insert(0, top_dir)
from aigames import *
import torch.multiprocessing as mp
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def self_play(game_class, agent, n_games: int):
    logging.debug('Inside self-play worker.')
    sys.stdout.flush()
    for _ in range(n_games):
        cur_game = game_class([agent, agent])
        logging.debug('Playing game...')
        cur_game.play()


def evaluation_worker(model, device, evaluation_queue, results_queues, kill_queue):
    logging.debug('Inside evaluation worker.')
    worker = MultiprocessingAlphaEvaluationWorker(model, device, evaluation_queue, results_queues, kill_queue)
    worker.evaluate_until_killed()


def training_worker(model, optimizer, device, train_queue, kill_queue):
    logging.debug('Inside training worker.')
    sys.stdout.flush()
    worker = MultiprocessingAlphaTrainingWorker(model, optimizer, device, train_queue, kill_queue)
    worker.train_until_killed()


def train(game_class, n_self_play_procs, n_games, device='cpu',
          n_evaluation_workers=1, n_training_workers=1,
          evaluation_queue_max_size=100, train_queue_max_size=100,
          training_tau=1, c_puct=1):
    mp.set_start_method('forkserver')
    evaluation_queue = mp.Queue(maxsize=evaluation_queue_max_size)
    train_queue = mp.Queue(maxsize=train_queue_max_size)
    results_queues = [mp.Queue(maxsize=1) for _ in range(n_self_play_procs)]
    kill_queue = mp.Queue(maxsize=1)

    model = AlphaAgentNetwork1(game_class)
    optimizer = torch.optim.Adam(model.parameters())

    evaluation_processes = []
    for i in range(n_evaluation_workers):
        logging.debug('Starting evaluation worker...')
        cur_evaluation_process = mp.Process(target=evaluation_worker, args=(model, device, evaluation_queue, results_queues, kill_queue))
        cur_evaluation_process.start()
        evaluation_processes.append(cur_evaluation_process)

    training_processes = []
    for i in range(n_training_workers):
        logging.debug('Starting training worker...')
        cur_training_process = mp.Process(target=training_worker, args=(model, optimizer, device, train_queue, kill_queue))
        cur_training_process.start()
        training_processes.append(cur_training_process)

    self_play_processes = []
    for i in range(n_self_play_procs):
        logging.debug('Starting self-play worker...')
        cur_evaluator = MultiprocessingAlphaEvaluator(i, model, evaluation_queue, results_queues[i], train_queue)
        cur_agent = AlphaAgent(game_class, cur_evaluator, training_tau, c_puct)
        cur_process = mp.Process(target=self_play, args=(game_class, cur_agent, n_games))
        cur_process.start()
        self_play_processes.append(cur_process)


    sys.stdout.flush()

    for sc in self_play_processes:
        sc.join()

    kill_queue.put(True)
    for sc in evaluation_processes:
        sc.join()
    for sc in training_processes:
        sc.join()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--n_self_play_procs', type=int, help='Number of self-play processes')
    parser.add_argument('-n', '--n_games', type=int, help='Number of self-play games per process')
    args = parser.parse_args()

    train(TicTacToe, args.n_self_play_procs, args.n_games)

if __name__ == '__main__':
    main()
