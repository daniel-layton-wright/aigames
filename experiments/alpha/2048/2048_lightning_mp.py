import argparse
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from aigames.agent.alpha_agent import TrainingTau
from aigames.training_manager.alpha_training_manager_mp import AlphaTrainingRunLightningMP,\
    AlphaTrainingHyperparametersLightningMP, AlphaTrainingRunSelfPlayMetadataListener,\
    AlphaTrainingRunSelfPlayMetadataProcessor
from aigames.game.twenty_forty_eight import TwentyFortyEight
from .network_architectures import TwentyFortyEightNetwork, TwentyFortyEightEvaluator
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from aigames.utils.utils import get_all_slots, add_all_slots_to_arg_parser
import os
import torch.multiprocessing as mp
import torch


class TwentyFortyEightBestTileListener(AlphaTrainingRunSelfPlayMetadataListener):
    def __init__(self):
        super().__init__()
        self.best_tile = None

    def on_game_end(self, game):
        self.best_tile = game.state.grid.max()
        self.metadata_queue.put({'best_tile': self.best_tile})


class TwentyFortyEightBestTileProcessor(AlphaTrainingRunSelfPlayMetadataProcessor):
    def process_metadata(self, metadata, training_run: AlphaTrainingRunLightningMP):
        # Loop over metadata and if the item is a dict and has a best tile entry, take the average
        best_tiles = []
        for item in metadata:
            if isinstance(item, dict) and 'best_tile' in item:
                best_tiles.append(item['best_tile'])

        if len(best_tiles) > 0:
            avg_best_tile = sum(best_tiles) / len(best_tiles)
        else:
            avg_best_tile = None

        # Log to the training run experiment
        training_run.logger.experiment.log({'best_tile': avg_best_tile})


def main():
    mp.set_start_method('forkserver')
    network = TwentyFortyEightNetwork(n_blocks=2, n_channels=64, n_out_channels=16)

    # Set up the evaluator. If a GPU is available, use it, otherwise use the CPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    evaluator = TwentyFortyEightEvaluator(network, device=device)
    hyperparams = AlphaTrainingHyperparametersLightningMP()
    hyperparams.self_play_every_n_epochs = 20
    hyperparams.n_self_play_games = 100
    hyperparams.n_self_play_procs = 4
    hyperparams.max_data_size = 100*42*2
    hyperparams.n_mcts = 100
    hyperparams.dirichlet_alpha = 1
    hyperparams.dirichlet_epsilon = 0.25
    hyperparams.lr = 0.1
    hyperparams.weight_decay = 0
    hyperparams.training_tau = TrainingTau(tau_schedule_list=[1 for _ in range(21)] + [0 for _ in range(21)])
    hyperparams.c_puct = 4
    hyperparams.batch_size = 64
    hyperparams.game_listeners = [TwentyFortyEightBestTileListener()]
    hyperparams.metadata_processors = [TwentyFortyEightBestTileProcessor()]

    # Set up an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    add_all_slots_to_arg_parser(parser, hyperparams)
    parser.add_argument('--ckpt_dir', type=str, default=f'./ckpt/2048/')
    parser.add_argument('--debug', '-d', action='store_true', help='Open PDB at the end')
    parser.add_argument('--max_epcohs', type=int, default=100, help='Max epochs')

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

    training_run = AlphaTrainingRunLightningMP(TwentyFortyEight, network, evaluator, hyperparams)
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
