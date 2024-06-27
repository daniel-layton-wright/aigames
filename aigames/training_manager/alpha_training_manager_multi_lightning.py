from typing import Tuple, Dict, Any, Union, Optional
from .alpha_training_manager import *
import pytorch_lightning as pl
import torch.utils.data
import math
from .alpha_dataset_multi import TrajectoryDataset, AlphaDatasetMulti
from .hyperparameters import AlphaMultiTrainingHyperparameters
from ..agent.alpha_agent_multi import AlphaAgentMulti, ConstantMCTSIters, AlphaAgentMultiListener
from ..game.game_multi import GameMulti
from ..utils.listeners import MaxActionGameKiller, ActionCounterProgressBar
from ..utils.utils import import_string


class AlphaMultiNetwork(nn.Module, BaseAlphaEvaluator):
    def loss(self, states, pis, values, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError()

    @property
    def device(self):
        return next(self.parameters()).device


class AlphaMultiTrainingRunLightning(pl.LightningModule):
    def __init__(self, game_class: Type[GameMulti], hyperparams: AlphaMultiTrainingHyperparameters,
                 agent_class=AlphaAgentMulti, dataset: Union[AlphaDatasetMulti, Type[AlphaDatasetMulti], None] = None):
        """

        :param game:
        :param network:
        """
        super().__init__()
        self.save_hyperparameters('game_class', 'hyperparams')
        self.hyperparams = hyperparams
        self.game_class = game_class
        self.network = import_string(hyperparams.network_class)(**hyperparams.network_args)
        self.network = torch.compile(self.network)
        self.dataset = self.create_dataset(dataset)
        self.agent_class = agent_class
        self.agent = agent_class(self.game_class, self.network, self.hyperparams, listeners=[self.dataset])
        self.game = self.game_class(self.hyperparams.n_parallel_games, self.agent, listeners=hyperparams.game_listeners)
        self.eval_game = self.game_class(self.hyperparams.n_parallel_eval_games, self.agent,
                                         listeners=hyperparams.eval_game_listeners)
        self.n_self_play_rounds = 0
        self.lr_save_tmp = []
        self.doing_dummy_epoch = False
        self.loaded_from_checkpoint = False
        self.doing_dummy_batch_before_resume = False
        self.resume_game = False

    def create_dataset(self, dataset: Union[AlphaDatasetMulti, Type[AlphaDatasetMulti], None]) -> AlphaDatasetMulti:
        # If dataset is a dataset object already just use it
        if isinstance(dataset, AlphaDatasetMulti):
            return dataset
        # Or if the dataset is a class initialize it
        elif isinstance(dataset, type):
            return dataset(self.network, self.hyperparams)
        # Or if dataset is None return a default one
        elif dataset is None:
            return TrajectoryDataset(self.network, self.hyperparams)
        else:
            raise ValueError(f'Unknown dataset type: {dataset}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hyperparams.lr,
                                     weight_decay=self.hyperparams.weight_decay)
        return [optimizer]

    def loss(self, processed_states, action_distns, values, *args):
        value_loss, distn_loss = self.network.loss(processed_states, action_distns, values, *args)

        mean_loss = self.hyperparams.value_weight_in_loss * value_loss + distn_loss
        return mean_loss, value_loss, distn_loss

    def on_fit_start(self):
        if len(self.dataset) == 0:
            # Use a dummy agent so that we don't interfere if a game is being reloaded
            dummy_agent = self.agent_class(self.game_class, self.network, self.hyperparams, listeners=[self.dataset])
            dummy_game = self.game_class(self.hyperparams.n_parallel_games, dummy_agent,
                                         listeners=[MaxActionGameKiller(2),
                                                    ActionCounterProgressBar(2, 'Dummy game to test backprop')])

            # play game for 2 moves then abort for minimal dataset for lightning to do its sanity checks then do the
            # full play in on_train_epoch_start
            if self.hyperparams.self_play_every_n_epochs > 0:
                # Do first self-play.
                dummy_agent.train()
                self.network.eval()
                dummy_game.play()

            # Put network back in train mode
            self.network.train()
            self.doing_dummy_epoch = True

    def self_play(self):
        if self.n_self_play_rounds in self.hyperparams.clear_dataset_before_self_play_rounds:
            self.dataset.clear()

        self.hyperparams.training_tau.update_metric('self_play_round', self.n_self_play_rounds)
        self.hyperparams.training_tau.update_metric('optimizer_step', self.trainer.global_step)
        self.log_to_wandb({'training_tau': self.hyperparams.training_tau.get_tau(0)})

        self.hyperparams.td_lambda.update_self_play_round(self.n_self_play_rounds)
        self.log_to_wandb({'td_lambda': self.hyperparams.td_lambda.get_lambda()})

        # Always put network in eval mode when playing games (for batch norm stuff)
        self.agent.train()
        self.network.eval()
        self.game.play()
        self.n_self_play_rounds += 1
        self.resume_game = False
        self.after_self_play_game()

    def log_to_wandb(self, log_dict):
        if self.logger is not None and hasattr(self.logger, 'experiment'):
            d = {
                **log_dict,
                'epoch': self.current_epoch,
                'trainer/global_step': self.global_step,
                'self_play_round': self.n_self_play_rounds
            }

            self.logger.experiment.log(d)

    def time_to_play_game(self):
        return (self.hyperparams.self_play_every_n_epochs > 0
                and self.current_epoch >= 0
                and self.current_epoch % self.hyperparams.self_play_every_n_epochs == 0)

    def time_to_play_eval_game(self):
        return (not self.resume_game and self.hyperparams.eval_game_every_n_epochs > 0
                and self.current_epoch > 0
                and self.current_epoch % self.hyperparams.eval_game_every_n_epochs == 0)

    def time_to_play_eval_game_network_only(self):
        return (not self.resume_game and self.hyperparams.eval_game_network_only_every_n_epochs > 0
                and self.current_epoch >= 0
                and self.current_epoch % self.hyperparams.eval_game_network_only_every_n_epochs == 0)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        """
        If this was loaded from a checkpoint and we need to resume a game, then we just do a single batch to make
        sure back prop is working.

        If we do not need to resume a game, then we train as normal on reloaded dataset.
        """
        if self.doing_dummy_epoch and batch_idx > 0:
            return -1

        if self.resume_game and self.doing_dummy_batch_before_resume and batch_idx > 0:
            # If we loaded from a checkpoint mid-game, then we already trained on this data, do one step to check
            # backprop is working then skip epoch
            return -1

    def on_train_epoch_start(self) -> None:
        if self.doing_dummy_epoch:
            # Set learning rate to 0 temporarily
            for opt in self.trainer.optimizers:
                for param_group in opt.param_groups:
                    self.lr_save_tmp.append(param_group['lr'])
                    param_group['lr'] = 0.0

    def on_train_epoch_end(self) -> None:
        # Save model checkpoint here, before self plays
        if not self.doing_dummy_epoch and not self.doing_dummy_batch_before_resume:
            for checkpoint_callback in self.trainer.checkpoint_callbacks:
                checkpoint_callback.on_train_epoch_end(self.trainer, self)

        if self.time_to_play_eval_game_network_only():
            # Temporarily set n_iters to 0 so we just use the network result
            tmp = self.agent.hyperparams.n_mcts_iters
            self.agent.hyperparams.n_mcts_iters = ConstantMCTSIters(0)
            self.agent.eval()
            self.network.eval()
            self.eval_game.play()
            self.after_eval_game_network_only()
            self.agent.hyperparams.n_mcts_iters = tmp

        if self.time_to_play_eval_game():
            self.agent.eval()
            self.network.eval()
            self.eval_game.play()
            self.after_eval_game()

        if self.time_to_play_game():
            # TODO: see if something weird is going on here when loading from checkpoint
            # Clear the dummy epoch only when we're going to play a real game right after to fill the dataset
            if self.doing_dummy_epoch:
                for opt in self.trainer.optimizers:
                    # Set learning rate back
                    for param_group in opt.param_groups:
                        param_group['lr'] = self.lr_save_tmp.pop()

                self.hyperparams.training_tau.update_metric('avg_total_num_moves', None)
                self.dataset.clear()
                self.doing_dummy_epoch = False  # Done with dummy epoch

            self.self_play()
            self.doing_dummy_batch_before_resume = False

        self.agent.train()
        self.network.train()

        self.log('dataset/size', self.dataset.num_datapoints)
        self.log('dataset/total_datapoints_seen', self.dataset.total_datapoints_seen)

    def after_self_play_game(self):
        pass

    def after_eval_game(self):
        pass

    def after_eval_game_network_only(self):
        pass

    def training_step(self, batch, nb_batch) -> dict:
        if self.doing_dummy_epoch:
            self.network.eval()  # don't update batchnorm statistics from dummy data

        loss, value_loss, distn_loss = self.loss(*batch)

        self.log('loss/loss', loss)
        self.log('loss/value_loss', value_loss)
        self.log('loss/distn_loss', distn_loss)
        return loss

    def train_dataloader(self):
        return self.dataset

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.hyperparams.save_dataset_in_checkpoint and not self.doing_dummy_epoch:
            checkpoint['dataset'] = self.dataset

        # TODO: this writes a checkpoint without the game if a checkpoint was laoded without a dataset but with a game
        if (~self.game.is_term).any():
            # Game is still going, save to checkpoint
            checkpoint['game'] = self.game

        checkpoint['n_self_play_rounds'] = self.n_self_play_rounds

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if 'dataset' in checkpoint:
            self.set_dataset(checkpoint['dataset'])

        if 'n_self_play_rounds' in checkpoint:
            self.n_self_play_rounds = checkpoint['n_self_play_rounds']

        if 'game' in checkpoint:
            self.game: GameMulti = checkpoint['game']
            self.agent.episode_history = self.game.player.episode_history
            self.agent.move_number_in_current_game = self.game.player.move_number_in_current_game
            self.agent.game = self.game
            self.game.player = self.agent
            self.game.listeners = self.hyperparams.game_listeners
            self.game.n_parallel_games = self.hyperparams.n_parallel_games

            self.game.to(self.hyperparams.device)
            self.agent.to(self.hyperparams.device)

            self.resume_game = True
            self.doing_dummy_batch_before_resume = True

        self.dataset.evaluator = self.network
        self.loaded_from_checkpoint = True

    def set_dataset(self, dataset: AlphaDatasetMulti):
        # Remove old dataset from agent
        self.agent.listeners.remove(self.dataset)

        dataset.evaluator = self.network
        self.dataset = dataset
        self.dataset.hyperparams = self.hyperparams
        self.dataset.enforce_max_size()  # in case this has been changed on the restart
        self.dataset.to(self.hyperparams.dataset_device)
        self.agent.listeners.append(self.dataset)
