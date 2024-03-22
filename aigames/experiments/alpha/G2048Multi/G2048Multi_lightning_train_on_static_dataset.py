import argparse
import sys

import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from ....agent.alpha_agent_multi import TrainingTau
from ....training_manager.alpha_training_manager_multi_lightning import AlphaMultiTrainingRunLightning
from aigames.training_manager.hyperparameters import AlphaMultiTrainingHyperparameters
from ....utils.listeners import ActionCounterProgressBar
from .network_architectures import G2048MultiNetwork, G2048MultiNetworkV2
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from ....utils.utils import add_all_slots_to_arg_parser, load_from_arg_parser
import os
from ....game.G2048_multi import get_G2048Multi_game_class
from .G2048Multi_lightning import G2048TrainingRun


def main():
    # Set up an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=False)
    parser.add_argument('--debug', '-d', action='store_true', help='Open PDB at the end')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max epochs')

    # Parse the args and set the hyperparams
    args, _ = parser.parse_known_args(sys.argv[1:])

    training_run = G2048TrainingRun.load_from_checkpoint(args.ckpt_path)

    add_all_slots_to_arg_parser(parser, training_run.hyperparams)
    args = parser.parse_args()
    load_from_arg_parser(args, training_run.hyperparams)

    # Start wandb run
    wandb_kwargs = dict(
        project='aigames2',
        group='G2048Multi_lightning',
        reinit=False,
    )

    # So we can get the name for model checkpointing
    wandb.init(**wandb_kwargs)
    wandb_run = wandb.run.name or os.path.split(wandb.run.path)[-1]

    training_run.hyperparams.self_play_every_n_epochs = -1
    training_run.hyperparams.eval_game_network_only_every_n_epochs = -1
    training_run.hyperparams.eval_game_every_n_epochs = -1
    training_run.hyperparams.max_data_size = 10
    training_run.dataset.enforce_max_size()

    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=0,
                         logger=pl_loggers.WandbLogger(**wandb_kwargs), log_every_n_steps=1, max_epochs=args.max_epochs)

    trainer.fit(training_run)

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
