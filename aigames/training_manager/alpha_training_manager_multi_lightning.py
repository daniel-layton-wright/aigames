from dataclasses import dataclass, field
from .alpha_training_manager import *
import pytorch_lightning as pl
import torch.utils.data
import math
from .alpha_training_manager_multi import BasicAlphaDatasetMulti, TensorDataset
from ..agent.alpha_agent_multi import AlphaAgentHyperparametersMulti, AlphaAgentMulti
from ..game.game_multi import GameMulti


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


class BasicAlphaDatasetLightning(BasicAlphaDatasetMulti):
    def __init__(self, evaluator: BaseAlphaEvaluator = None,
                 hyperparams: AlphaMultiTrainingHyperparameters = None, process_state=True):
        if hyperparams is None:
            hyperparams = AlphaMultiTrainingHyperparameters()

        super().__init__(evaluator, hyperparams.max_data_size, process_state, hyperparams.min_data_size)
        self.hyperparams = hyperparams

    def __iter__(self):
        dataset = TensorDataset(self.states, self.pis, self.rewards)
        sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=self.hyperparams.batch_size)
        return iter(dataloader)

    def __len__(self):
        n = self.states.shape[0] if self.states is not None else 0
        return math.ceil(n / self.hyperparams.batch_size)


class AlphaMultiTrainingRunLightning(pl.LightningModule):
    def __init__(self, game_class: Type[GameMulti], network: nn.Module, evaluator: BaseAlphaEvaluator,
                 hyperparams: AlphaMultiTrainingHyperparameters):
        """

        :param game:
        :param network: Pass in a callable that returns a network
        :param optimizer: Pass in a function that will do f( network ) -> optimizer
        :param training_listeners:
        """
        super().__init__()
        self.save_hyperparameters()
        self.hyperparams = hyperparams
        self.game_class = game_class
        self.network = network
        self.alpha_evaluator = copy.deepcopy(evaluator)
        self.alpha_evaluator.network = self.network
        self.dataset = BasicAlphaDatasetLightning(self.alpha_evaluator, self.hyperparams)
        self.agent = AlphaAgentMulti(self.game_class, self.alpha_evaluator, self.hyperparams, listeners=[self.dataset])
        self.game = self.game_class(self.hyperparams.n_parallel_games, self.agent, listeners=hyperparams.game_listeners)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hyperparams.lr,
                                     weight_decay=self.hyperparams.weight_decay)
        return [optimizer]

    def loss(self, processed_states, action_distns, values):
        pred_distns, pred_values = self.network(processed_states)
        losses = (self.hyperparams.value_weight_in_loss * ((values - pred_values) ** 2).sum(dim=1)
                  - torch.sum(action_distns * torch.log(pred_distns), dim=1))
        mean_loss = losses.mean()
        return mean_loss

    def on_fit_start(self):
        # Do first self-play
        self.network.eval()
        self.game.play()

    def on_train_batch_start(self, batch, batch_idx: int):
        pass  # In the parent class, it does self play here

    def on_train_start(self):
        pass  # In the parent class, it does self play here

    def on_train_epoch_start(self) -> None:
        if self.current_epoch > 0 and self.current_epoch % self.hyperparams.self_play_every_n_epochs == 0:
            self.network.eval()
            self.game.play()

        self.network.train()

    def on_train_epoch_end(self) -> None:
        pass

    def training_step(self, batch, nb_batch) -> dict:
        loss = self.loss(*batch)
        self.log('loss', loss)
        return loss

    def train_dataloader(self):
        return self.dataset
