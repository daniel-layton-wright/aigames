import argparse
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from aigames.agent.alpha_agent import TrainingTau
from aigames.training_manager.alpha_training_manager_mp import AlphaTrainingRunLightningMP,\
    AlphaTrainingHyperparametersLightningMP
from aigames.game.twenty_forty_eight import TwentyFortyEight
from .network_architectures import TwentyFortyEightNetwork, TwentyFortyEightEvaluator
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from aigames.utils.utils import get_all_slots, add_all_slots_to_arg_parser
import os
import torch.multiprocessing as mp


def main():
    mp.set_start_method('forkserver')
    network = TwentyFortyEightNetwork(n_blocks=2, n_channels=64, n_out_channels=16)
    evaluator = TwentyFortyEightEvaluator(network)
    hyperparams = AlphaTrainingHyperparametersLightningMP()
    hyperparams.self_play_every_n_epochs = 100
    hyperparams.n_self_play_games = 1
    hyperparams.n_self_play_procs = 4
    hyperparams.max_data_size = 100*42*2
    hyperparams.n_mcts = 100
    hyperparams.dirichlet_alpha = 1
    hyperparams.dirichlet_epsilon = 0.25
    hyperparams.lr = 0.001
    hyperparams.weight_decay = 0
    hyperparams.training_tau = TrainingTau(tau_schedule_list=[1 for _ in range(21)] + [0 for _ in range(21)])
    hyperparams.c_puct = 4
    hyperparams.batch_size = 64

    # Set up an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    add_all_slots_to_arg_parser(parser, hyperparams)
    parser.add_argument('--ckpt_dir', type=str, default=f'./ckpt/2048/')
    parser.add_argument('--debug', '-d', action='store_true', help='Open PDB at the end')

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
                         callbacks=[model_checkpoint], log_every_n_steps=1, max_epochs=99)
    trainer.fit(training_run)

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
