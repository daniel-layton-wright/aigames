import os

import pytorch_lightning as pl
from pytorch_lightning import Callback
import torch

from aigames.game.game_multi import GameListenerMulti


class EpisodeHistoryCheckpoint(Callback):
    def __init__(self, ckpt_dir, file_name='latest_episode_history.pkl'):
        self.ckpt_dir = ckpt_dir
        self.file_name = file_name

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # Save episode history for debugging purposes
        torch.save(pl_module.agent.episode_history, os.path.join(self.ckpt_dir, self.file_name))


class CheckpointMidGame(Callback, GameListenerMulti):
    def __init__(self, save_every_n_moves=100):
        super().__init__()
        self.cur_move = 0
        self.save_every_n_moves = save_every_n_moves
        self.trainer = None
        self.pl_module = None

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self.trainer = trainer
        self.pl_module = pl_module

    def before_game_start(self, game):
        self.cur_move = 0

    def after_action(self, game):
        self.cur_move += 1

        if self.cur_move % self.save_every_n_moves == 0:
            for checkpoint in self.trainer.checkpoint_callbacks:
                if isinstance(checkpoint, ModelCheckpoint):
                    monitor_candidates = checkpoint._monitor_candidates(self.trainer)
                    checkpoint._save_last_checkpoint(self.trainer, monitor_candidates)

    def __getstate__(self):
        return {'save_every_n_moves': self.save_every_n_moves, 'cur_move': self.cur_move}

    def __json__(self):
        return {'save_every_n_moves': self.save_every_n_moves, 'cur_move': self.cur_move}


class GameProgressCallback(Callback, GameListenerMulti):
    """
    A training callback which will log the current move number in the training game being played by the trainer
    """
    def __init__(self, log_name='training_game/move_number'):
        super().__init__()
        self.cur_move = 0
        self.trainer = None
        self.pl_module = None
        self.log_name = log_name

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self.trainer = trainer
        self.pl_module = pl_module

    def before_game_start(self, game):
        self.cur_move = 0

    def on_action(self, game, actions):
        self.cur_move += 1
        self.pl_module.logger.experiment.log({self.log_name: self.cur_move})

    def on_game_end(self, game):
        self.cur_move = 0
        self.pl_module.logger.experiment.log({self.log_name: self.cur_move})
