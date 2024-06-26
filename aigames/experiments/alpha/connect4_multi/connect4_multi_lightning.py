import argparse
from collections import defaultdict
from typing import List, Any, Tuple
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from aigames.game.connect4_multi import get_connect4_multi_class
from aigames.mcts.mcts import UCBFormulaType
from ....agent.alpha_agent_multi import TrainingTau, TDLambdaByRound, ConstantMCTSIters
from ....training_manager.alpha_training_manager_multi_lightning import AlphaMultiTrainingRunLightning
from aigames.training_manager.hyperparameters import AlphaMultiTrainingHyperparameters
from ....utils.listeners import ActionCounterProgressBar, RewardListenerMulti, PerGameActionCounter
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from ....utils.utils import add_all_slots_to_arg_parser, load_from_arg_parser, import_string
import os
from ....game.game_multi import GameListenerMulti
import sys
import torch


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


class TrainingTauStepSchedule(TrainingTau):
    """
    A training tau based on a schedule for the optimizer step number

    Example: TrainingTauStepSchedule([(1.0, int(100e3)), (0.5, int(200e3)), (0.1, int(300e3)), (0.0, None)])
    """
    def __init__(self, schedule: List[Tuple[float, int]]):
        super().__init__(0)
        self.schedule = schedule
        self.i = 0

    def update_metric(self, key, val):
        if key == 'optimizer_step':
            self.i = next((i for i, (_, step) in enumerate(self.schedule) if (step is None or step > val)),
                          len(self.schedule) - 1  # default
                          )

    def get_tau(self, move_number):
        return self.schedule[self.i][0]


class Connect4TrainingRun(AlphaMultiTrainingRunLightning):
    def log_game_results(self, game, suffix):
        log_dict = {}

        undiscounted_reward_listener = next(filter(lambda x: isinstance(x, RewardListenerMulti) and x.discount == 1.,
                                                   game.listeners), None)

        discounted_reward_listener = next(filter(lambda x: isinstance(x, RewardListenerMulti) and x.discount != 1.,
                                                 game.listeners), None)

        per_game_action_counter = next(filter(lambda x: isinstance(x, PerGameActionCounter), game.listeners), None)

        if undiscounted_reward_listener is not None:
            log_dict[f'avg_undiscounted_reward0/{suffix}'] = undiscounted_reward_listener.rewards[:, 0].mean().item()

        if discounted_reward_listener is not None:
            log_dict[f'avg_discounted_reward0/{suffix}'] = discounted_reward_listener.rewards[: 0].mean().item()

        if per_game_action_counter is not None:
            log_dict[f'avg_game_length/{suffix}'] = per_game_action_counter.i.float().mean()

        # Log this
        self.log_to_wandb(log_dict)

    def after_self_play_game(self):
        self.log_game_results(self.game, 'train')

    def after_eval_game(self):
        self.log_game_results(self.eval_game, 'eval')

    def after_eval_game_network_only(self):
        self.log_game_results(self.eval_game, 'eval_network_only')
        self.hyperparams.training_tau.update_metric('eval_game_avg_max_tile',
                                                    self.eval_game.states.amax(dim=(1, 2)).float().mean().item())

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.network.eval()

        # Log value of simple states: a beginning state, a good intermediate state, and a state probably close to losing
        example_states = torch.FloatTensor(
             [[[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]]]
        )

        pi, val = self.network.evaluate(example_states.to(self.hyperparams.device))

        for i in range(len(example_states)):
            self.log(f'example_state{i}/value0', val[i, 0].detach().item())
            self.log(f'example_state{i}/policy3', pi[i].flatten()[3].item())

        self.network.train()


def main():
    # Set up an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt_path', type=str, default=None, help='Path to a checkpoint to restore from')
    parser.add_argument('--network_class', type=str,
                        default='aigames.experiments.alpha.connect4_multi.network_architectures.Connect4Network')

    sysargv = sys.argv[1:]
    if '--help' in sysargv:
        sysargv.remove('--help')
    ckpt_path_args, _ = parser.parse_known_args(sysargv)

    checkpoint_mid_game = CheckpointMidGame()

    if ckpt_path_args.restore_ckpt_path is not None:
        training_run = Connect4TrainingRun.load_from_checkpoint(ckpt_path_args.restore_ckpt_path, map_location='cpu')
        hyperparams = training_run.hyperparams
    else:
        hyperparams = AlphaMultiTrainingHyperparameters()
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
        hyperparams.c_puct = 1.25
        hyperparams.lr = 0.0003
        hyperparams.weight_decay = 1e-5
        hyperparams.td_lambda = TDLambdaByRound([1, 0.9, 0.8, 0.7, 0.6, 0.5])
        hyperparams.training_tau = TrainingTauStepSchedule([(1.0, int(200e3)), (0.5, int(400e3)),
                                                            (0.1, int(600e3)), (0.0, None)])
        hyperparams.batch_size = 1024
        hyperparams.data_buffer_full_size = 16_384  # stabilize things by doing 16 steps before using new network for next TD estimates
        hyperparams.discount = 0.999
        hyperparams.game_listeners = [ActionCounterProgressBar(42, description='Train game action count'),
                                      RewardListenerMulti(1), RewardListenerMulti(hyperparams.discount),
                                      PerGameActionCounter()]
        hyperparams.eval_game_listeners = [ActionCounterProgressBar(42, description='Eval game action count'),
                                           RewardListenerMulti(1), RewardListenerMulti(hyperparams.discount),
                                           PerGameActionCounter()]
        hyperparams.ucb_formula = UCBFormulaType.MuZeroLog
        hyperparams.save_dataset_in_checkpoint = True
        hyperparams.clear_dataset_before_self_play_rounds = []

        network_class = import_string(ckpt_path_args.network_class)
        if hasattr(network_class, 'add_args_to_arg_parser'):
            network_class.add_args_to_arg_parser(parser)

    add_all_slots_to_arg_parser(parser, hyperparams)
    parser.add_argument('--ckpt_dir', type=str, default=f'./ckpt/connect4_multi/')
    parser.add_argument('--debug', '-d', action='store_true', help='Open PDB at the end')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Max epochs')
    parser.add_argument('--restore_wandb_run_id', type=str, default=None)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)

    # Parse the args and set the hyperparams
    args = parser.parse_args()
    load_from_arg_parser(args, hyperparams)

    Connect4Multi = get_connect4_multi_class(hyperparams.device)

    # remove the CheckpointMidGame if it exists and add the current one
    for x in hyperparams.game_listeners:
        if isinstance(x, CheckpointMidGame):
            hyperparams.game_listeners.remove(x)

    hyperparams.game_listeners.append(checkpoint_mid_game)

    if ckpt_path_args.restore_ckpt_path is None:
        if hasattr(network_class, 'init_from_arg_parser'):
            network = network_class.init_from_arg_parser(args)
        else:
            network = network_class()

        training_run = Connect4TrainingRun(Connect4Multi, network, hyperparams,
                                           dataset=import_string(hyperparams.dataset_class))
    else:
        training_run = Connect4TrainingRun.load_from_checkpoint(ckpt_path_args.restore_ckpt_path,
                                                             hyperparams=hyperparams,
                                                             map_location=hyperparams.device)

    # Start wandb run
    wandb_kwargs = dict(
        project='aigames2',
        group='connect4_lightning',
        reinit=False,
    )

    if args.restore_wandb_run_id is not None:
        wandb_kwargs['id'] = args.restore_wandb_run_id
        wandb_kwargs['resume'] = True

    # So we can get the name for model checkpointing
    wandb.init(**wandb_kwargs)
    wandb_run = wandb.run.name or os.path.split(wandb.run.path)[-1]
    wandb.run.watch(training_run.network)

    torch.autograd.set_detect_anomaly(True)

    model_checkpoint = ModelCheckpoint(dirpath=os.path.join(args.ckpt_dir, wandb_run), save_last=True)
    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=1,
                         logger=pl_loggers.WandbLogger(**wandb_kwargs),
                         callbacks=[model_checkpoint, checkpoint_mid_game], log_every_n_steps=1,
                         max_epochs=args.max_epochs,
                         gradient_clip_val=args.gradient_clip_val
                         )
    trainer.fit(training_run, ckpt_path=args.restore_ckpt_path)

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
