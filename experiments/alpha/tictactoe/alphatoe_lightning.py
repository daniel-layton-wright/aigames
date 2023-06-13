from aigames.agent.alpha_agent import *
from aigames.training_manager.alpha_training_manager_lightning import *
from aigames.game.tictactoe import *
from aigames import Flatten
import torch
import pytorch_lightning as pl


class TicTacToeNetwork(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.n_channels1 = 64
        self.base_out_features = 16

        self.base = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=self.n_channels1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.n_channels1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.n_channels1, out_channels=self.base_out_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.base_out_features),
            nn.ReLU(),
            Flatten()
        )

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=9),
            nn.Softmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=1)
        )

    def forward(self, processed_state):
        base = self.base(processed_state)

        policy = self.policy_head(base).squeeze()
        value = self.value_head(base)
        return policy, value


class FastTicTacToeEvaluator(AlphaNetworkEvaluator):
    def process_state(self, state: FastTicTacToeState):
        s = state.tensor_state
        if abs(s).sum() % 2 == 1:
            s = copy.deepcopy(s)
            s *= -1

        t = torch.zeros(2, 3, 3, dtype=torch.float)
        t[0, :, :] = s

        t[1, :, :] = (t[0, :, :] == -1)
        t[0, :, :] += t[1, :, :]

        return t


def main():
    network = TicTacToeNetwork()
    evaluator = FastTicTacToeEvaluator(network)
    hyperparams = AlphaTrainingHyperparameters()
    hyperparams.min_data_size = 256
    hyperparams.n_mcts = 100
    hyperparams.dirichlet_alpha = 0.3
    hyperparams.dirichlet_epsilon = 0.5
    hyperparams.lr = 0.001
    hyperparams.weight_decay = 1e-4
    hyperparams.training_tau = TrainingTau(0.5)
    hyperparams.c_puct = 5
    hyperparams.batch_size = 64

    training_run = AlphaTrainingRunLightning(FastTicTacToe, evaluator, hyperparams)

    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=1, logger=pl.loggers.WandbLogger())
    trainer.fit(training_run)


if __name__ == '__main__':
    main()
