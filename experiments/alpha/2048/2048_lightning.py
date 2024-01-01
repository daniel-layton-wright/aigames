import argparse
from typing import Type
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from aigames.agent.alpha_agent import AlphaAgent, TrainingTau
from aigames.base import SequentialGame
from aigames.training_manager.alpha_training_manager import AlphaNetworkEvaluator
from aigames.training_manager.alpha_training_manager_lightning import AlphaTrainingRunLightning, AlphaTrainingHyperparametersLightning
from aigames.game.twenty_forty_eight import TwentyFortyEight
from .network_architectures import TwentyFortyEightNetwork, TwentyFortyEightEvaluator
import torch
import pytorch_lightning as pl
from aigames.utils.utils import get_all_slots
import os
import copy
import torch.nn as nn


class AlphaTrainingRunLightningConnect4(AlphaTrainingRunLightning):
    """
    Need this class for the eval function specific to Connect4
    """

    def __init__(self, game_class: Type[SequentialGame], network: nn.Module, alpha_evaluator: AlphaNetworkEvaluator,
                 hyperparams: AlphaTrainingHyperparametersLightning):
        super().__init__(game_class, network, alpha_evaluator, hyperparams)
        self.avg_score = None

    def on_train_epoch_end(self) -> None:
        return
        agent = self.agents[0]


def main():
    network = TwentyFortyEightNetwork()
    evaluator = TwentyFortyEightEvaluator(network)
    hyperparams = AlphaTrainingHyperparametersLightning()
    hyperparams.min_data_size = 20
    hyperparams.max_data_size = 500*32*2
    hyperparams.n_mcts = 100
    hyperparams.dirichlet_alpha = 1
    hyperparams.dirichlet_epsilon = 0.25
    hyperparams.lr = 0.01
    hyperparams.weight_decay = 1e-4
    hyperparams.training_tau = TrainingTau(1.0)
    hyperparams.c_puct = 1
    hyperparams.batch_size = 32
    hyperparams.play_game_every_n_iters = 20
    hyperparams.num_samples_per_epoch = 20000

    # Setup an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default=f'./ckpt/2048/')
    for slot in get_all_slots(hyperparams):
        parser.add_argument(f'--{slot}', type=type(getattr(hyperparams, slot)), default=getattr(hyperparams, slot))

    # Parse the args and set the hyperparams
    args = parser.parse_args()
    for slot in get_all_slots(hyperparams):
        setattr(hyperparams, slot, getattr(args, slot))

    # Start wandb run
    wandb_kwargs = dict(
        project='aigames2',
        group='2048_lightning',
        reinit=False,
    )

    # So we can get the name for model checkpointing
    wandb.init(**wandb_kwargs)
    wandb_run = wandb.run.name or os.path.split(wandb.run.path)[-1]

    training_run = AlphaTrainingRunLightning(TwentyFortyEight, network, evaluator, hyperparams)
    model_checkpoint = ModelCheckpoint(dirpath=os.path.join(args.ckpt_dir, wandb_run), save_last=True)
    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=1, logger=pl.loggers.WandbLogger(**wandb_kwargs),
                         callbacks=[model_checkpoint])
    trainer.fit(training_run)


if __name__ == '__main__':
    main()
