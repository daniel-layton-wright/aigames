import argparse
import sys
from collections import defaultdict
from ctypes import c_int
from dataclasses import dataclass
from typing import List
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from .... import ListDataset
from ....agent.alpha_agent_multi import TrainingTau
from ....training_manager.alpha_training_manager_mp import AlphaEvaluatorMP
from ....training_manager.alpha_training_manager_multi_lightning import AlphaMultiTrainingRunLightning, \
    AlphaMultiTrainingHyperparameters, BasicAlphaDatasetLightning
from ....utils.listeners import ActionCounterProgressBar
from .network_architectures import G2048MultiNetwork, G2048MultiNetworkV2
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from ....utils.utils import add_all_slots_to_arg_parser, load_from_arg_parser
import os
from ....game.G2048_multi import get_G2048Multi_game_class
from ....game.game_multi import GameListenerMulti, GameMulti
from .G2048Multi_lightning import G2048TrainingRun, TrainingTauDecreaseOnPlateau, import_string
import torch.multiprocessing as mp
import torch
import queue


@dataclass(kw_only=True, slots=True)
class AlphaMultiTrainingHyperparametersMP(AlphaMultiTrainingHyperparameters):
    n_self_play_procs: int = 4
    n_data_loader_workers: int = 1


class AlphaTrainingMPData:
    def __init__(self):
        self.stop = mp.Value(c_int)


class AlphaSelfPlayMP:
    def __init__(self, game: GameMulti):
        self.game = game

    def self_play_loop(self, response_queue_index: int, mp_data: AlphaTrainingMPData):
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.game.player.evaluator.response_queue_i = response_queue_index

        # Play at least one game
        self.game.play()


class BasicAlphaDatasetLightningMP(BasicAlphaDatasetLightning):
    def __init__(self, evaluator, hyperparams: AlphaMultiTrainingHyperparametersMP, process_state=True):
        super().__init__(evaluator, hyperparams, process_state)
        self.queue_for_data_to_add = mp.Queue()  # Queue for data to add to the dataset
        self.hyperparams = hyperparams

    def __iter__(self):
        dataset = ListDataset(self.states, self.pis, self.rewards)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.hyperparams.batch_size, shuffle=True,
                                                 num_workers=self.hyperparams.n_data_loader_workers)
        return iter(dataloader)

    def on_data_point(self, state, pi, reward):
        self.queue_for_data_to_add.put((state, pi, reward))

    def add_data_from_queue(self):
        while True:
            try:
                state, pi, reward = self.queue_for_data_to_add.get(block=False)
            except queue.Empty:
                return
            else:
                super().on_data_point(state, pi, reward)


class G2048TrainingRunMP(G2048TrainingRun):
    def __init__(self, game_class, evaluator, hyperparams: AlphaMultiTrainingHyperparametersMP):
        super().__init__(game_class, evaluator, hyperparams)
        self.hyperparams = hyperparams  # Just to shut PyCharm up

        # Update the evaluator to use the multiprocessing version
        self.evaluator = AlphaEvaluatorMP(self, n_procs=hyperparams.n_self_play_procs)
        self.agent.alpha_evaluator = self.evaluator

        # Update the dataset to use the multiprocessing version
        self.agent.listeners.remove(self.dataset)
        self.dataset = BasicAlphaDatasetLightningMP(self.network, hyperparams)
        self.agent.listeners.append(self.dataset)

    def on_fit_start(self):
        super().on_fit_start()

        if len(self.dataset) == 0:
            self.dataset.add_data_from_queue()

    def self_play(self):
        torch.multiprocessing.set_sharing_strategy('file_system')
        # This makes sure the agent collects data:
        self.agent.train()

        # When doing self-play, we are not training the network, only generating training data
        # And we need the network in eval mode for, e.g., batch-norm
        self.network.eval()

        self_play_processes = []

        mp_data = AlphaTrainingMPData()
        self_play_mp = AlphaSelfPlayMP(self.game)

        eval_proc = mp.Process(target=self.evaluator.evaluation_loop, args=(mp_data,))
        eval_proc.start()

        for i in range(self.hyperparams.n_self_play_procs):
            proc = mp.Process(target=self_play_mp.self_play_loop, args=(i, mp_data))
            proc.start()
            self_play_processes.append(proc)

        while len(self_play_processes) > 0:
            proc = self_play_processes[0]
            proc.join(timeout=1)

            if proc.exitcode is None:  # The process is still running
                self.dataset.add_data_from_queue()  # Need to empty the data queue to be able to join process
            else:  # The process finished
                self_play_processes.remove(proc)

        self.dataset.add_data_from_queue()

        mp_data.stop.value = 2
        eval_proc.join()


def main():
    # Set up an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt_path', type=str, default=None, help='Path to a checkpoint to restore from')
    sysargv = sys.argv[1:]
    if '--help' in sysargv:
        sysargv.remove('--help')
    ckpt_path_args, _ = parser.parse_known_args(sysargv)

    # game_progress_callback = GameProgressCallback()

    if ckpt_path_args.restore_ckpt_path is not None:
        training_run = G2048TrainingRunMP.load_from_checkpoint(ckpt_path_args.restore_ckpt_path, map_location='cpu')
        hyperparams = training_run.hyperparams
    else:
        hyperparams = AlphaMultiTrainingHyperparametersMP()
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
        hyperparams.training_tau = TrainingTauDecreaseOnPlateau([1.0, 0.7, 0.5, 0.3, 0.1, 0.0],
                                                                'eval_game_avg_max_tile', 2)
        hyperparams.batch_size = 1024
        hyperparams.game_listeners = [ActionCounterProgressBar(1000, description='Train game action count'),
                                      # game_progress_callback
                                      ]
        hyperparams.eval_game_listeners = [ActionCounterProgressBar(1000, description='Eval game action count')]
        hyperparams.discount = 0.999
        hyperparams.clear_dataset_before_self_play_rounds = []

    add_all_slots_to_arg_parser(parser, hyperparams)
    parser.add_argument('--ckpt_dir', type=str, default=f'./ckpt/G2048Multi/')
    parser.add_argument('--debug', '-d', action='store_true', help='Open PDB at the end')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--network_class', type=str, default='aigames.experiments.alpha.G2048Multi.network_architectures.G2048MultiNetworkV2')
    parser.add_argument('--restore_wandb_run_id', type=str, default=None)

    # Parse the args and set the hyperparams
    args = parser.parse_args()
    load_from_arg_parser(args, hyperparams)

    G2048Multi = get_G2048Multi_game_class(hyperparams.device)

    if ckpt_path_args.restore_ckpt_path is None:
        network_class = import_string(args.network_class)
        network = network_class()
        training_run = G2048TrainingRunMP(G2048Multi, network, hyperparams)
    else:
        training_run = G2048TrainingRunMP.load_from_checkpoint(ckpt_path_args.restore_ckpt_path,
                                                             hyperparams=hyperparams,
                                                             map_location='cpu')

    # Start wandb run
    wandb_kwargs = dict(
        project='aigames2',
        group='G2048Multi_lightning',
        reinit=False,
    )

    if args.restore_wandb_run_id is not None:
        wandb_kwargs['id'] = args.restore_wandb_run_id
        wandb_kwargs['resume'] = True

    # So we can get the name for model checkpointing
    wandb.init(**wandb_kwargs)
    wandb_run = wandb.run.name or os.path.split(wandb.run.path)[-1]

    model_checkpoint = ModelCheckpoint(dirpath=os.path.join(args.ckpt_dir, wandb_run), save_last=True)
    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=1,  # hyperparams.self_play_every_n_epochs,
                         logger=pl_loggers.WandbLogger(**wandb_kwargs),
                         callbacks=[model_checkpoint,
                                    # game_progress_callback
                                    ], log_every_n_steps=1,
                         max_epochs=args.max_epochs,
                         )
    trainer.fit(training_run, ckpt_path=args.restore_ckpt_path)

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()

