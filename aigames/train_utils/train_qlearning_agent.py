import torch.nn as nn
import torch
import os
from .monitor import QLearningMonitor
from aigames import QLearningAgent
from .google_cloud import google_cloud_upload_file


def train_qlearning_agent(game_class, qlearning_agent: QLearningAgent, monitor: QLearningMonitor=QLearningMonitor(),
                          start_iter: int=0, n_games: int=100000):

    monitor.before_training_start(qlearning_agent)
    players = [qlearning_agent for _ in range(game_class.N_PLAYERS)]

    i = start_iter
    while i < n_games:
        game = game_class(players, verbose=False, monitor=monitor.before_player_action)
        game.play()

        i += 1

        monitor.on_game_end(qlearning_agent, i)


def save_checkpoint(job_dir, game_number, qlearning_agent, filename):
    if job_dir is None:
        return
    else:
        torch.save({
            'game_number': game_number,
            'model_state_dict': qlearning_agent.Q.state_dict(),
            'optimizer_state_dict': qlearning_agent.optimizer.state_dict(),
            'loss_history': qlearning_agent.loss_history,
            'loss_ema_history': qlearning_agent.loss_ema_history,
            'replay_memory': qlearning_agent.replay_memory
        }, filename)
        google_cloud_upload_file(job_dir, filename)


def load_checkpoint(checkpoint_path, qlearning_agent):
    start_iter = 0
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        qlearning_agent.Q.load_state_dict(checkpoint['model_state_dict'])
        qlearning_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        qlearning_agent.loss_history = checkpoint['loss_history']
        qlearning_agent.loss_ema_history = checkpoint['loss_ema_history']
        qlearning_agent.replay_memory = checkpoint['replay_memory']
        start_iter = checkpoint['iter']

    return start_iter
