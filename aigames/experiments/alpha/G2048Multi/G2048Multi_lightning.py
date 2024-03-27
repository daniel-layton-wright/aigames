import argparse
from collections import defaultdict
from typing import List, Any, Tuple
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from aigames.mcts.mcts import UCBFormulaType
from ....agent.alpha_agent_multi import TrainingTau, TDLambdaByRound, ConstantMCTSIters
from ....training_manager.alpha_training_manager_multi_lightning import AlphaMultiTrainingRunLightning
from aigames.training_manager.hyperparameters import AlphaMultiTrainingHyperparameters
from ....utils.listeners import ActionCounterProgressBar
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from ....utils.utils import add_all_slots_to_arg_parser, load_from_arg_parser, import_string
import os
from ....game.G2048_multi import get_G2048Multi_game_class
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


class TrainingTauDecreaseOnPlateau(TrainingTau):
    def __init__(self, tau_schedule: List[float], plateau_metric, plateau_patience,
                 max_optimizer_steps_before_tau_decrease: int = -1):
        super().__init__(0)
        self.tau_schedule = tau_schedule
        self.i = 0
        self.metrics = defaultdict(list)
        self.plateau_metric = plateau_metric
        self.plateau_patience = plateau_patience
        self.j = -1
        self.max_j = -1
        self.max_metric = None
        self.max_optimizer_steps_before_tau_decrease = max_optimizer_steps_before_tau_decrease
        self.last_self_optimizer_step_tau_decrease = 0
        self.optimizer_step = 0

    def get_tau(self, move_number):
        return self.tau_schedule[self.i]

    def backwards_compatible_check(self):
        if not hasattr(self, 'optimizer_step'):
            self.optimizer_step = 0
            self.last_self_optimizer_step_tau_decrease = 0
            self.max_optimizer_steps_before_tau_decrease = -1

    def update_metric(self, key, val):
        self.backwards_compatible_check()

        if key == 'optimizer_step':
            self.optimizer_step = val

            if 0 < self.max_optimizer_steps_before_tau_decrease <= val - self.last_self_optimizer_step_tau_decrease:
                self.i = min(self.i + 1, len(self.tau_schedule) - 1)
                self.max_metric = None
                self.max_j = self.j
                self.last_self_optimizer_step_tau_decrease = val

        if key != self.plateau_metric:
            return

        self.j += 1

        if self.max_metric is None or val > self.max_metric:
            self.max_metric = val
            self.max_j = self.j

        if self.j - self.max_j >= self.plateau_patience:
            self.i = min(self.i + 1, len(self.tau_schedule) - 1)
            self.max_metric = None
            self.max_j = self.j
            self.last_self_optimizer_step_tau_decrease = self.optimizer_step


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


class EpisodeHistoryCheckpoint(Callback):
    def __init__(self, ckpt_dir, file_name='latest_episode_history.pkl'):
        self.ckpt_dir = ckpt_dir
        self.file_name = file_name

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # Save episode history for debugging purposes
        torch.save(pl_module.agent.episode_history, os.path.join(self.ckpt_dir, self.file_name))


class G2048TrainingRun(AlphaMultiTrainingRunLightning):
    def log_game_results(self, game, suffix):
        max_tiles = game.states.amax(dim=(1, 2))
        avg_max_tile = max_tiles.mean()
        max_best_tile = max_tiles.amax()

        fraction_reached_1024 = (max_tiles >= 10).float().mean()
        fraction_reached_2048 = (max_tiles >= 11).float().mean()

        # Log this
        self.log_to_wandb({
            f'avg_best_tile/{suffix}': avg_max_tile,
            f'fraction_reached_1024/{suffix}': fraction_reached_1024,
            f'fraction_reached_2048/{suffix}': fraction_reached_2048,
            f'max_best_tile/{suffix}': max_best_tile
        })

    def after_self_play_game(self):
        self.log_game_results(self.game, 'train')

        # Log fraction of dataset that has highest tile in corner
        kernel = -4 * torch.ones(15, 16, 4, 4)
        for i in range(1, 16):
            kernel[i - 1, :(i + 1)] = 0
            kernel[i - 1, i, 0, 0] = 1
            kernel[i - 1, i, 0, 3] = 1
            kernel[i - 1, i, 3, 0] = 1
            kernel[i - 1, i, 3, 3] = 1

        s = self.dataset.states()
        conv = torch.nn.functional.conv2d(s, kernel.to(s.device), padding=0, stride=1)
        conv = conv.amax(dim=(1, 2, 3))
        fraction_max_in_corner = (conv > 0).float().mean().item()
        self.log('dataset/fraction_max_in_corner', fraction_max_in_corner)

        # Compute average discounted and un-discounted reward from just the last round
        r = self.dataset.rewards(device='cpu').squeeze()[-self.hyperparams.n_parallel_games:]
        undiscounted_reward = r.sum(dim=1).mean().item()
        discounted_reward = ((self.hyperparams.discount ** torch.arange(0, r.shape[1])).reshape(1, -1) * r).sum(dim=1)
        discounted_reward = discounted_reward.mean().item()
        self.log('avg_undiscounted_reward/train', undiscounted_reward)
        self.log('avg_discounted_reward/train', discounted_reward)

    def after_eval_game(self):
        self.log_game_results(self.eval_game, 'eval')

    def after_eval_game_network_only(self):
        self.log_game_results(self.eval_game, 'eval_network_only')
        self.hyperparams.training_tau.update_metric('eval_game_avg_max_tile',
                                                    self.eval_game.states.amax(dim=(1, 2)).mean().item())

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

        pi, val = self.network.evaluate(example_states.to(self.hyperparams.device))

        for i in range(len(example_states)):
            self.log(f'example_state{i}/value', val[i].detach().item())
            self.log(f'example_state{i}/policy_left', pi[i].flatten()[0].item())

        self.network.train()


def main():
    # Set up an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt_path', type=str, default=None, help='Path to a checkpoint to restore from')
    parser.add_argument('--network_class', type=str,
                        default='aigames.experiments.alpha.G2048Multi.network_architectures.G2048MultiNetworkV2')

    sysargv = sys.argv[1:]
    if '--help' in sysargv:
        sysargv.remove('--help')
    ckpt_path_args, _ = parser.parse_known_args(sysargv)

    checkpoint_mid_game = CheckpointMidGame()

    if ckpt_path_args.restore_ckpt_path is not None:
        training_run = G2048TrainingRun.load_from_checkpoint(ckpt_path_args.restore_ckpt_path, map_location='cpu')
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
        hyperparams.c_puct = 1.25  # Can be low/normal when scaleQ is True
        hyperparams.lr = 0.0003
        hyperparams.weight_decay = 1e-5
        hyperparams.td_lambda = TDLambdaByRound([1, 0.9, 0.8, 0.7, 0.6, 0.5])
        hyperparams.training_tau = TrainingTauStepSchedule([(1.0, int(25e3)), (0.5, int(50e3)),
                                                            (0.1, int(75e3)), (0.0, None)])
        hyperparams.batch_size = 1024
        hyperparams.data_buffer_full_size = 16_384  # stabilize things by doing 16 steps before using new network for next TD estimates
        hyperparams.game_listeners = [ActionCounterProgressBar(1500, description='Train game action count')]
        hyperparams.eval_game_listeners = [ActionCounterProgressBar(1500, description='Eval game action count')]
        hyperparams.discount = 0.999
        hyperparams.ucb_formula = UCBFormulaType.MuZeroLog
        hyperparams.save_dataset_in_checkpoint = True
        hyperparams.clear_dataset_before_self_play_rounds = []

        network_class = import_string(ckpt_path_args.network_class)
        if hasattr(network_class, 'add_args_to_arg_parser'):
            network_class.add_args_to_arg_parser(parser)

    add_all_slots_to_arg_parser(parser, hyperparams)
    parser.add_argument('--ckpt_dir', type=str, default=f'./ckpt/G2048Multi/')
    parser.add_argument('--debug', '-d', action='store_true', help='Open PDB at the end')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Max epochs')
    parser.add_argument('--restore_wandb_run_id', type=str, default=None)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)

    # Parse the args and set the hyperparams
    args = parser.parse_args()
    load_from_arg_parser(args, hyperparams)

    G2048Multi = get_G2048Multi_game_class(hyperparams.device)

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

        training_run = G2048TrainingRun(G2048Multi, network, hyperparams, dataset=hyperparams.dataset_type)
    else:
        training_run = G2048TrainingRun.load_from_checkpoint(ckpt_path_args.restore_ckpt_path,
                                                             hyperparams=hyperparams,
                                                             map_location=hyperparams.device)

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
