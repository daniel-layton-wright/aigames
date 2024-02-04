import argparse
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

from aigames import CommandLineGame
from aigames.agent.alpha_agent import TrainingTau
from aigames.game.game_multi import GameListenerMulti
from aigames.training_manager.alpha_training_manager_multi_lightning import AlphaMultiTrainingRunLightning, \
    AlphaMultiTrainingHyperparameters
from aigames.utils.listeners import ActionCounterProgressBar
from .network_architectures import G2048MultiNetwork, G2048MultiEvaluator
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from aigames.utils.utils import get_all_slots, add_all_slots_to_arg_parser
import os
import torch
from aigames.game.G2048_multi import get_G2048Multi_game_class


class G2048BestTileListener(GameListenerMulti):
    def __init__(self, training_run):
        super().__init__()
        self.training_run = training_run

    def on_game_end(self, game):
        max_tiles = game.states.amax(dim=(1, 2))
        avg_max_tile = max_tiles.mean()

        # Log this
        self.training_run.logger.experiment.log({'best_tile_train': avg_max_tile})


def main():
    network = G2048MultiNetwork(n_blocks=2, n_channels=64, n_out_channels=16)

    hyperparams = AlphaMultiTrainingHyperparameters()
    hyperparams.self_play_every_n_epochs = 10
    hyperparams.n_parallel_games = 1000
    hyperparams.max_data_size = 200000
    hyperparams.min_data_size = 512
    hyperparams.mcts_hyperparams.n_iters = 250
    hyperparams.mcts_hyperparams.dirichlet_alpha = 1.0
    hyperparams.mcts_hyperparams.dirichlet_epsilon = 0.25
    hyperparams.mcts_hyperparams.c_puct = 250
    hyperparams.lr = 0.1
    hyperparams.weight_decay = 1e-4
    hyperparams.training_tau = TrainingTau(fixed_tau_value=1)
    hyperparams.batch_size = 512
    hyperparams.game_listeners = [ActionCounterProgressBar(500)]

    # Set up an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    add_all_slots_to_arg_parser(parser, hyperparams)
    add_all_slots_to_arg_parser(parser, hyperparams.mcts_hyperparams)
    parser.add_argument('--ckpt_dir', type=str, default=f'./ckpt/G2048Multi/')
    parser.add_argument('--debug', '-d', action='store_true', help='Open PDB at the end')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max epochs')

    # Parse the args and set the hyperparams
    args = parser.parse_args()
    for slot in get_all_slots(hyperparams):
        setattr(hyperparams, slot, getattr(args, slot))

    for slot in get_all_slots(hyperparams.mcts_hyperparams):
        setattr(hyperparams.mcts_hyperparams, slot, getattr(args, slot))

    # Start wandb run
    wandb_kwargs = dict(
        project='aigames2',
        group='G2048Multi_lightning',
        reinit=False,
    )

    # So we can get the name for model checkpointing
    wandb.init(**wandb_kwargs)
    wandb_run = wandb.run.name or os.path.split(wandb.run.path)[-1]

    evaluator = G2048MultiEvaluator(network, device=hyperparams.device)
    G2048Multi = get_G2048Multi_game_class(hyperparams.device)

    training_run = AlphaMultiTrainingRunLightning(G2048Multi, network, evaluator, hyperparams)
    training_run.game.listeners.append(G2048BestTileListener(training_run))
    model_checkpoint = ModelCheckpoint(dirpath=os.path.join(args.ckpt_dir, wandb_run), save_last=True)
    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=hyperparams.self_play_every_n_epochs,
                         logger=pl_loggers.WandbLogger(**wandb_kwargs),
                         callbacks=[model_checkpoint], log_every_n_steps=1, max_epochs=args.max_epochs)
    trainer.fit(training_run)

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
