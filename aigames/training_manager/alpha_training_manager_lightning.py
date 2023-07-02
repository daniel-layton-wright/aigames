import copy
from .alpha_training_manager import *
import pytorch_lightning as pl
import torch.utils.data
import math
from aigames.game.tictactoe import FastTicTacToe
from aigames.agent.minimax_agent import MinimaxAgent
from aigames.utils.utils import play_tournament


class AlphaTrainingHyperparametersLightning(AlphaTrainingHyperparameters):
    __slots__ = ['num_samples_per_epoch', 'play_game_every_n_iters']

    def __init__(self):
        super().__init__()
        self.num_samples_per_epoch = 100  # this is used by the dataset
        self.play_game_every_n_iters = 1


class BasicAlphaDatasetLightning(BasicAlphaDataset):
    def __init__(self, evaluator: BaseAlphaEvaluator = None,
                 hyperparams: AlphaTrainingHyperparametersLightning = None, process_state=True):
        if hyperparams is None:
            hyperparams = AlphaTrainingHyperparameters()

        super().__init__(evaluator, hyperparams.max_data_size, process_state, hyperparams.min_data_size)
        self.hyperparams = hyperparams

    def __iter__(self):
        dataset = ListDataset(self.states, self.pis, self.rewards)
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True,
                                                 num_samples=(self.hyperparams.num_samples_per_epoch *
                                                              self.hyperparams.batch_size))
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=self.hyperparams.batch_size)
        return iter(dataloader)

    def __len__(self):
        return math.ceil(len(self.states) / self.hyperparams.batch_size)


class AlphaTrainingRunLightning(pl.LightningModule):
    def __init__(self, game_class: Type[SequentialGame], network: nn.Module, evaluator: BaseAlphaEvaluator,
                 hyperparams: AlphaTrainingHyperparametersLightning):
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
        self.agent = AlphaAgent(self.game_class, self.alpha_evaluator, self.hyperparams, listeners=[self.dataset])
        self.game = self.game_class([self.agent for _ in range(self.game_class.get_n_players())])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hyperparams.lr,
                                     weight_decay=self.hyperparams.weight_decay)
        return [optimizer]

    def loss(self, processed_states, action_distns, values):
        pred_distns, pred_values = self.network(processed_states)
        losses = (values - pred_values) ** 2 - torch.sum(action_distns * torch.log(pred_distns), dim=1)
        mean_loss = losses.mean()
        return mean_loss

    def on_train_start(self):  # TODO: this might need to be on_fit_start
        with tqdm(total=self.hyperparams.min_data_size) as pbar:
            pbar.set_description('Generating initial dataset')
            while len(self.dataset) < self.hyperparams.min_data_size:
                self.game.play()
                pbar.update(len(self.dataset) - pbar.n)  # Because update expects the increment (not cumulative total)

    def on_train_epoch_start(self) -> None:
        for agent in self.agents:  # Make sure to put agents in training mode
            agent.train()

    def on_train_batch_start(self, batch, batch_idx: int):
        if batch_idx % self.hyperparams.play_game_every_n_iters == 0:
            self.game.play()

    def training_step(self, batch, nb_batch) -> dict:
        loss = self.loss(*batch)
        self.log('loss', loss)
        return loss

    def train_dataloader(self):
        return self.dataset
