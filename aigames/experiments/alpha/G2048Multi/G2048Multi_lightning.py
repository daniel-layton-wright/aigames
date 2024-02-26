import argparse
from collections import defaultdict
from typing import List

import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from ....agent.alpha_agent_multi import TrainingTau
from ....training_manager.alpha_training_manager_multi_lightning import AlphaMultiTrainingRunLightning, \
    AlphaMultiTrainingHyperparameters
from ....utils.listeners import ActionCounterProgressBar
from .network_architectures import G2048MultiNetwork, G2048MultiNetworkV2
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from ....utils.utils import add_all_slots_to_arg_parser, load_from_arg_parser
import os
from ....game.G2048_multi import get_G2048Multi_game_class
from ....game.game_multi import GameListenerMulti
import sys


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
        self.pl_module.log(self.log_name, self.cur_move)


class TrainingTauDecreaseOnPlateau(TrainingTau):
    def __init__(self, tau_schedule: List[float], plateau_metric, plateau_patience):
        super().__init__(0)
        self.tau_schedule = tau_schedule
        self.i = 0
        self.metrics = defaultdict(list)
        self.plateau_metric = plateau_metric
        self.plateau_patience = plateau_patience
        self.j = -1
        self.max_j = -1
        self.max_metric = None

    def get_tau(self, move_number):
        return self.tau_schedule[min(self.i, len(self.tau_schedule) - 1)]

    def update_metric(self, key, val):
        if key != self.plateau_metric:
            return

        self.j += 1

        if self.max_metric is None or val > self.max_metric:
            self.max_metric = val
            self.max_j = self.j

        if self.j - self.max_j >= self.plateau_patience:
            self.i += 1
            self.max_metric = None
            self.max_j = self.j


class G2048TrainingRun(AlphaMultiTrainingRunLightning):
    def log_game_results(self, game, suffix):
        max_tiles = game.states.amax(dim=(1, 2))
        avg_max_tile = max_tiles.mean()

        fraction_reached_1024 = (max_tiles >= 10).float().mean()
        fraction_reached_2048 = (max_tiles >= 11).float().mean()

        # Log this
        self.logger.experiment.log({f'best_tile/{suffix}': avg_max_tile})
        self.logger.experiment.log({f'fraction_reached_1024/{suffix}': fraction_reached_1024})
        self.logger.experiment.log({f'fraction_reached_2048/{suffix}': fraction_reached_2048})

    def on_fit_start(self):
        super().on_fit_start()
        self.log_game_results(self.game, 'train')

    def after_self_play_game(self):
        self.log_game_results(self.game, 'train')

    def after_eval_game(self):
        self.log_game_results(self.eval_game, 'eval')

    def after_eval_game_network_only(self):
        self.log_game_results(self.eval_game, 'eval_network_only')
        self.hyperparams.training_tau.update_metric('eval_game_avg_max_tile',
                                                    self.eval_game.states.amax(dim=(1, 2)).mean().item())


def main():
    # Set up an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt_path', type=str, default=None, help='Path to a checkpoint to restore from')
    ckpt_path_args, _ = parser.parse_known_args(sys.argv[1:])

    if ckpt_path_args.restore_ckpt_path is not None:
        training_run = G2048TrainingRun.load_from_checkpoint(ckpt_path_args.restore_ckpt_path, map_location='cpu')
        hyperparams = training_run.hyperparams
    else:
        hyperparams = AlphaMultiTrainingHyperparameters()
        hyperparams.self_play_every_n_epochs = 10
        hyperparams.n_parallel_games = 1000
        hyperparams.max_data_size = 4000000
        hyperparams.min_data_size = 1024
        hyperparams.n_mcts_iters = 100
        hyperparams.dirichlet_alpha = 0.25
        hyperparams.dirichlet_epsilon = 0.1
        hyperparams.scaleQ = True
        hyperparams.c_puct = 2  # Can be low/normal when scaleQ is True
        hyperparams.lr = 0.002
        hyperparams.weight_decay = 1e-5
        hyperparams.training_tau = TrainingTauDecreaseOnPlateau([1, 0.7, 0.5, 0.3, 0.1, 0],
                                                                'eval_game_avg_max_tile', 2)
        hyperparams.batch_size = 1024
        hyperparams.game_listeners = [ActionCounterProgressBar(1000, description='Train game action count')]
        hyperparams.eval_game_listeners = [ActionCounterProgressBar(1000, description='Eval game action count')]
        hyperparams.discount = 0.999
        hyperparams.clear_dataset_before_self_play_rounds = []

        parser.add_argument('--network_class', type=str, default='G2048MultiNetworkV2')

    add_all_slots_to_arg_parser(parser, hyperparams)
    parser.add_argument('--ckpt_dir', type=str, default=f'./ckpt/G2048Multi/')
    parser.add_argument('--debug', '-d', action='store_true', help='Open PDB at the end')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max epochs')

    # Parse the args and set the hyperparams
    args = parser.parse_args()
    load_from_arg_parser(args, hyperparams)

    # Start wandb run
    wandb_kwargs = dict(
        project='aigames2',
        group='G2048Multi_lightning',
        reinit=False,
    )

    # So we can get the name for model checkpointing
    wandb.init(**wandb_kwargs)
    wandb_run = wandb.run.name or os.path.split(wandb.run.path)[-1]

    G2048Multi = get_G2048Multi_game_class(hyperparams.device)

    if ckpt_path_args.restore_ckpt_path is None:
        network_class = getattr(sys.modules[__name__], args.network_class)
        network = network_class()
        training_run = G2048TrainingRun(G2048Multi, network, hyperparams)

    model_checkpoint = ModelCheckpoint(dirpath=os.path.join(args.ckpt_dir, wandb_run), save_last=True)
    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=1,  # hyperparams.self_play_every_n_epochs,
                         logger=pl_loggers.WandbLogger(**wandb_kwargs),
                         callbacks=[model_checkpoint], log_every_n_steps=1,
                         max_epochs=args.max_epochs)
    trainer.fit(training_run)

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
