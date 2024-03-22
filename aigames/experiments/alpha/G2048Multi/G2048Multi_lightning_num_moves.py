import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Any, Tuple
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from aigames.agent.alpha_agent_multi_adaptive import AlphaAgentMultiAdaptive
from aigames.training_manager.alpha_dataset_multi import NumMovesTrajectoryDataset
from ....agent.alpha_agent_multi import TrainingTau, TDLambdaByRound, TDLambda, AlphaAgentHyperparametersMulti, \
    ConstantMCTSIters
from ....training_manager.alpha_training_manager_multi_lightning import AlphaMultiTrainingRunLightning
from aigames.training_manager.hyperparameters import AlphaMultiTrainingHyperparameters
from ....utils.listeners import ActionCounterProgressBar
from .network_architectures import G2048MultiNetwork, G2048MultiNetworkV2
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from ....utils.utils import add_all_slots_to_arg_parser, load_from_arg_parser
import os
from ....game.G2048_multi import get_G2048Multi_game_class
from ....game.game_multi import GameListenerMulti
import sys
from importlib import import_module
import torch
from .G2048Multi_lightning import import_string, G2048TrainingRun, EpisodeHistoryCheckpoint, TrainingTauDecreaseOnPlateau


class AdaptiveTrainingTauDecreaseOnPlateau(TrainingTauDecreaseOnPlateau):
    def __init__(self, tau_schedule: List[Tuple[float, float]], plateau_metric, plateau_patience,
                 max_optimizer_steps_before_tau_decrease):
        super().__init__(tau_schedule, plateau_metric, plateau_patience, max_optimizer_steps_before_tau_decrease)
        self.tau_schedule = tau_schedule
        self.avg_total_num_moves = None
        self.expected_num_moves = None

    def get_tau(self, move_number):
        if self.avg_total_num_moves is None or self.expected_num_moves is None:
            return self.tau_schedule[self.i][0]
        else:
            # Interpolate between the tau values
            frac_remaining = torch.maximum(self.expected_num_moves / self.avg_total_num_moves, torch.zeros_like(self.expected_num_moves))
            frac_remaining = torch.minimum(frac_remaining, torch.ones_like(frac_remaining))

            tau = (self.tau_schedule[self.i][0] * frac_remaining +
                   + self.tau_schedule[self.i][1] * (1 - frac_remaining))

            return tau

    def update_metric(self, key, val):
        super().update_metric(key, val)

        if key == 'avg_total_num_moves':
            self.avg_total_num_moves = val

        if key == 'expected_num_moves':
            self.expected_num_moves = val


@dataclass(kw_only=True, slots=True)
class AlphaAgentMultiAdaptiveHyperparameters(AlphaMultiTrainingHyperparameters, AlphaAgentHyperparametersMulti):
    num_moves_weight_in_loss: float = 1


class G2048TrainingRunNumMoves(G2048TrainingRun):
    def __init__(self, game_class, network, hyperparams: AlphaAgentMultiAdaptiveHyperparameters):
        dataset = NumMovesTrajectoryDataset(network, hyperparams)
        super().__init__(game_class, network, hyperparams, dataset=dataset, agent_class=AlphaAgentMultiAdaptive)
        self.hyperparams = hyperparams

    def loss(self, processed_states, action_distns, values, num_moves):
        value_loss, distn_loss, num_moves_loss = self.network.loss(processed_states, action_distns, values, num_moves)
        mean_loss = (distn_loss
                     + self.hyperparams.value_weight_in_loss * value_loss
                     + self.hyperparams.num_moves_weight_in_loss * num_moves_loss)
        return mean_loss, value_loss, distn_loss, num_moves_loss

    def training_step(self, batch, nb_batch) -> dict:
        if self.doing_dummy_epoch:
            self.network.eval()

        loss, value_loss, distn_loss, num_moves_loss = self.loss(*batch)
        self.log('loss/loss', loss)
        self.log('loss/value_loss', value_loss)
        self.log('loss/distn_loss', distn_loss)
        self.log('loss/num_moves_loss', num_moves_loss)
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.network.eval()

        # Log value of simple states: a beginning state, a good intermediate state, and a state probably close to losing
        example_states = torch.FloatTensor(
             [[[0, 0, 1, 0],
               [0, 0, 0, 0],
               [1, 0, 0, 0],
               [0, 0, 0, 0]],

              [[2, 1, 0, 0],
               [5, 4, 0, 0],
               [6, 5, 1, 0],
               [7, 6, 5, 2]],

              [[3, 1, 4, 0],
               [1, 5, 3, 0],
               [6, 3, 1, 5],
               [1, 10, 5, 1]]]
        )

        pi, val, num_moves = self.network.evaluate(example_states.to(self.hyperparams.device))

        for i in range(len(example_states)):
            self.log(f'example_state{i}/value', val[i].detach().item())
            self.log(f'example_state{i}/policy_left', pi[i].flatten()[0].item())
            self.log(f'example_state{i}/num_moves', num_moves[i].detach().item())

        self.network.train()


def main():
    # Set up an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt_path', type=str, default=None, help='Path to a checkpoint to restore from')
    parser.add_argument('--network_class', type=str, default='aigames.experiments.alpha.G2048Multi.network_architectures.G2048MultiNetworkV3')

    sysargv = sys.argv[1:]
    if '--help' in sysargv:
        sysargv.remove('--help')
    ckpt_path_args, _ = parser.parse_known_args(sysargv)

    # game_progress_callback = GameProgressCallback()

    if ckpt_path_args.restore_ckpt_path is not None:
        training_run = G2048TrainingRunNumMoves.load_from_checkpoint(ckpt_path_args.restore_ckpt_path, map_location='cpu')
        hyperparams = training_run.hyperparams
    else:
        hyperparams = AlphaAgentMultiAdaptiveHyperparameters()
        hyperparams.self_play_every_n_epochs = 1
        hyperparams.eval_game_every_n_epochs = 100
        hyperparams.eval_game_network_only_every_n_epochs = 1
        hyperparams.n_parallel_games = 1000
        hyperparams.max_data_size = 3500000
        hyperparams.min_data_size = 1024
        hyperparams.n_mcts_iters = ConstantMCTSIters(100)
        hyperparams.dirichlet_alpha = 0.25
        hyperparams.dirichlet_epsilon = 0.1
        hyperparams.scaleQ = True
        hyperparams.c_puct = 2  # Can be low/normal when scaleQ is True
        hyperparams.lr = 0.0005
        hyperparams.weight_decay = 1e-5
        hyperparams.training_tau = AdaptiveTrainingTauDecreaseOnPlateau(
            [(1.0, 0.5), (0.7, 0.2), (0.5, 0.1), (0.3, 0.0), (0.1, 0.0), (0.0, 0.0)],
            'eval_game_avg_max_tile', 4 * hyperparams.self_play_every_n_epochs,
            max_optimizer_steps_before_tau_decrease=50_000
        )
        hyperparams.td_lambda = TDLambda(0.5)
        hyperparams.num_moves_td_lambda = TDLambda(0.5)
        hyperparams.batch_size = 1024
        hyperparams.game_listeners = [ActionCounterProgressBar(1500, description='Train game action count'),
                                      ]
        hyperparams.eval_game_listeners = [ActionCounterProgressBar(1500, description='Eval game action count')]
        hyperparams.discount = 0.999
        hyperparams.clear_dataset_before_self_play_rounds = []
        hyperparams.num_moves_weight_in_loss = 0.5

        network_class = import_string(ckpt_path_args.network_class)
        if hasattr(network_class, 'add_args_to_arg_parser'):
            network_class.add_args_to_arg_parser(parser)

    add_all_slots_to_arg_parser(parser, hyperparams)
    parser.add_argument('--ckpt_dir', type=str, default=f'./ckpt/G2048Multi/')
    parser.add_argument('--debug', '-d', action='store_true', help='Open PDB at the end')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Max epochs')
    parser.add_argument('--restore_wandb_run_id', type=str, default=None)

    # Parse the args and set the hyperparams
    args = parser.parse_args()
    load_from_arg_parser(args, hyperparams)

    G2048Multi = get_G2048Multi_game_class(hyperparams.device)

    if ckpt_path_args.restore_ckpt_path is None:
        if hasattr(network_class, 'init_from_arg_parser'):
            network = network_class.init_from_arg_parser(args)
        else:
            network = network_class()

        training_run = G2048TrainingRunNumMoves(G2048Multi, network, hyperparams)
    else:
        training_run = G2048TrainingRunNumMoves.load_from_checkpoint(ckpt_path_args.restore_ckpt_path,
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
    episode_history_checkpoint = EpisodeHistoryCheckpoint(os.path.join(args.ckpt_dir, wandb_run))
    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=1,
                         logger=pl_loggers.WandbLogger(**wandb_kwargs),
                         callbacks=[model_checkpoint, episode_history_checkpoint
                                    ], log_every_n_steps=1,
                         max_epochs=args.max_epochs,
                         )
    trainer.fit(training_run, ckpt_path=args.restore_ckpt_path)

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
