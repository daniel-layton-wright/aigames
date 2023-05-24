from .alpha_training_manager import *
import pytorch_lightning as pl


class BasicAlphaDatasetLightning(BasicAlphaDataset):
    def __init__(self, evaluator: BaseAlphaEvaluator=None, max_size=50000, process_state=True, min_size=100,
                 batch_size=32):
        super().__init__(evaluator, max_size, process_state, min_size)
        self.batch_size = batch_size

    def __iter__(self):
        dataset = ListDataset(self.states, self.pis, self.rewards)
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=100*self.batch_size)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)
        return iter(dataloader)


class AlphaTrainingRunLightning(pl.LightningModule):
    def __init__(self, game_class: Type[SequentialGame], alpha_evaluator: AlphaNetworkEvaluator,
                 hyperparams: AlphaTrainingHyperparameters,
                 training_listeners: List[TrainingListener] = ()):
        """

        :param game:
        :param network: Pass in a callable that returns a network
        :param optimizer: Pass in a function that will do f( network ) -> optimizer
        :param training_listeners:
        """
        super().__init__()
        self.hyperparams = hyperparams
        self.save_hyperparameters('hyperparams')
        self.game_class = game_class
        self.alpha_evaluator = copy.deepcopy(alpha_evaluator)
        self.network = self.alpha_evaluator.network
        self.dataset = BasicAlphaDatasetLightning(self.alpha_evaluator, max_size=self.hyperparams.max_data_size)
        self.listeners = training_listeners
        self.agents = []
        self.n_iters = 0
        self.training_listeners = training_listeners

        for i in range(self.game_class.get_n_players()):
            cur_agent = AlphaAgent(self.game_class, self.alpha_evaluator, self.hyperparams,
                                   listeners=[self.dataset])
            self.agents.append(cur_agent)

        self.n_iters = 0
        self.game = self.game_class(self.agents, [*self.listeners])

        for listener in self.listeners:
            listener.before_begin_training(self)

        while len(self.dataset) < self.hyperparams.min_data_size:
            self.game.play()

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

    def training_step(self, batch, nb_batch) -> dict:
        self.game.play()
        loss = self.loss(*batch)
        self.log('loss', loss)
        return loss

    def train_dataloader(self):
        return self.dataset
