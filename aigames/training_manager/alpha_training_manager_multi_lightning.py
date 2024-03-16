from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Union
from .alpha_training_manager import *
import pytorch_lightning as pl
import torch.utils.data
import math
from .alpha_training_manager_multi import BasicAlphaDatasetMulti, TensorDataset, TrajectoryDataset
from ..agent.alpha_agent_multi import AlphaAgentHyperparametersMulti, AlphaAgentMulti, ConstantMCTSIters, \
    AlphaAgentMultiListener
from ..game.game_multi import GameMulti
from ..utils.listeners import MaxActionGameKiller


@dataclass(kw_only=True, slots=True)
class AlphaMultiTrainingHyperparameters(AlphaAgentHyperparametersMulti):
    n_parallel_games: int = 1000
    value_weight_in_loss: float = 1.0
    batch_size: int = 32
    device: str = 'cpu'
    game_listeners: list = field(default_factory=list)
    lr: float = 0.01
    weight_decay: float = 1e-5
    self_play_every_n_epochs: int = 10
    max_data_size: int = 10000000
    min_data_size: int = 1
    n_parallel_eval_games: int = 100
    eval_game_every_n_epochs: int = 10  # Set to -1 for never
    eval_game_network_only_every_n_epochs: int = 10
    eval_game_listeners: list = field(default_factory=list)
    clear_dataset_before_self_play_rounds: list = field(default_factory=list)
    save_dataset_in_checkpoint: bool = False
    data_buffer_full_size: int = 32
    td_truncate_length: int = 10


class BasicAlphaDatasetLightning(BasicAlphaDatasetMulti):
    def __init__(self, evaluator: BaseAlphaEvaluator = None,
                 hyperparams: AlphaMultiTrainingHyperparameters = None, process_state=True):
        if hyperparams is None:
            hyperparams = AlphaMultiTrainingHyperparameters()

        super().__init__(evaluator, hyperparams.max_data_size, process_state, hyperparams.min_data_size)
        self.hyperparams = hyperparams
        self.total_datapoints_seen = 0
        self.datapoints_seen_since_last_sample = 0

    def __iter__(self):
        dataset = TensorDataset(*(self.data[key] for key in self.data))
        sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=self.hyperparams.batch_size)
        return iter(dataloader)

    def __len__(self):
        n = self.data['states'].shape[0] if self.data['states'] is not None else 0
        return math.ceil(n / self.hyperparams.batch_size)

    def enforce_max_size(self):
        while self.data['states'] is not None and self.data['states'].shape[0] > self.hyperparams.max_data_size:
            self.pop()

    def on_data_point(self, states: torch.Tensor, pis: torch.Tensor, rewards: torch.Tensor, *args, **kwargs):
        super().on_data_point(states, pis, rewards, *args, **kwargs)
        self.total_datapoints_seen += states.shape[0]
        self.datapoints_seen_since_last_sample += states.shape[0]


class BasicAlphaDatasetMultiNumMoves(BasicAlphaDatasetLightning):
    def __init__(self, evaluator: BaseAlphaEvaluator = None, hyperparams: AlphaMultiTrainingHyperparameters = None,
                 process_state=True):
        super().__init__(evaluator, hyperparams, process_state)
        self.data['num_moves'] = None

    def get_data_names_and_values(self, states, pis, rewards, num_moves=None, *args, **kwargs):
        data = super().get_data_names_and_values(states, pis, rewards, *args, **kwargs)
        data.append(('num_moves', num_moves))
        return data


class AlphaMultiNetwork(nn.Module, BaseAlphaEvaluator):
    def loss(self, states, pis, values, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError()


class AlphaMultiTrainingRunLightning(pl.LightningModule):
    def __init__(self, game_class: Type[GameMulti], network: AlphaMultiNetwork,
                 hyperparams: AlphaMultiTrainingHyperparameters, agent_class=AlphaAgentMulti,
                 dataset: Union[AlphaAgentMultiListener, Type[AlphaAgentMultiListener], None] = None):
        """

        :param game:
        :param network: Pass in a callable that returns a network
        """
        super().__init__()
        self.save_hyperparameters('game_class', 'network', 'hyperparams')
        self.hyperparams = hyperparams
        self.game_class = game_class
        self.network = network
        self.dataset = self.create_dataset(dataset)
        self.agent = agent_class(self.game_class, self.network, self.hyperparams, listeners=[self.dataset])
        self.game = self.game_class(self.hyperparams.n_parallel_games, self.agent, listeners=hyperparams.game_listeners)
        self.eval_game = self.game_class(self.hyperparams.n_parallel_eval_games, self.agent,
                                         listeners=hyperparams.eval_game_listeners)
        self.n_self_play_rounds = 0
        self.lr_save_tmp = []
        self.doing_dummy_epoch = False

    # TODO : make base class for alpha dataset
    def create_dataset(self, dataset) -> AlphaAgentMultiListener:
        # If dataset is a dataset object already just use it
        if isinstance(dataset, AlphaAgentMultiListener):
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
            game_killer = MaxActionGameKiller(2)
            listeners_tmp = self.game.listeners
            self.game.listeners = [game_killer]

            # Idea: play game for 2 moves then abort for minimal dataset for lightning to do its sanity checks then do the
            # full play in on_train_epoch_start
            if self.hyperparams.self_play_every_n_epochs > 0:
                # Do first self-play.
                self.agent.train()
                self.network.eval()
                self.game.play()

            self.game.listeners = listeners_tmp

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
        return (self.hyperparams.eval_game_every_n_epochs > 0
                and self.current_epoch > 0
                and self.current_epoch % self.hyperparams.eval_game_every_n_epochs == 0)

    def time_to_play_eval_game_network_only(self):
        return (self.hyperparams.eval_game_network_only_every_n_epochs > 0
                and self.current_epoch >= 0
                and self.current_epoch % self.hyperparams.eval_game_network_only_every_n_epochs == 0)

    def on_train_epoch_start(self) -> None:
        if self.doing_dummy_epoch:
            # Set learning rate to 0 temporarily
            for opt in self.trainer.optimizers:
                for param_group in opt.param_groups:
                    self.lr_save_tmp.append(param_group['lr'])
                    param_group['lr'] = 0.0

    def on_train_epoch_end(self) -> None:
        # Save model checkpoint here, before self plays
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

        self.agent.train()
        self.network.train()

        self.log('dataset/size', self.dataset.num_states())
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

        checkpoint['n_self_play_rounds'] = self.n_self_play_rounds

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if 'dataset' in checkpoint:
            self.set_dataset(checkpoint['dataset'])

        if 'n_self_play_rounds' in checkpoint:
            self.n_self_play_rounds = checkpoint['n_self_play_rounds']

        self.dataset.evaluator = self.network

    def set_dataset(self, dataset):
        # Remove old dataset from agent
        self.agent.listeners.remove(self.dataset)

        dataset.evaluator = self.network
        self.dataset = dataset
        self.agent.listeners.append(self.dataset)
