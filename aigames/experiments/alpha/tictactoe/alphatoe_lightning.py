import argparse
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from aigames.agent.alpha_agent import *
from aigames.training_manager.alpha_training_manager_lightning import *
from aigames.game.tictactoe import *
from aigames import Flatten
import torch
import pytorch_lightning as pl
from aigames.utils.utils import get_all_slots


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


class AlphaTrainingRunLightningTTT(AlphaTrainingRunLightning):
    """
    Need this class for the eval function specific to TTT (most games can't eval against minimax)
    """
    def __init__(self, game_class: Type[SequentialGame], alpha_evaluator: AlphaNetworkEvaluator,
                 hyperparams: AlphaTrainingHyperparametersLightning):
        super().__init__(game_class, alpha_evaluator, hyperparams)
        self.avg_reward_against_minimax_ema = None
        self.minimax_agent = MinimaxAgent(FastTicTacToe)  # for eval

    def on_train_epoch_end(self) -> None:
        # Play tournament against minimax and log result
        agent = self.agents[-1]
        agent.eval()  # Put agent in eval mode so that it doesn't learn from these games (not fair to learn against minimax)
        avg_reward_against_minimax = play_tournament_old(FastTicTacToe, [self.minimax_agent, agent], 100, 1)
        agent.train()  # Put agent back in train mode

        self.log('avg_reward_against_minimax', avg_reward_against_minimax)

        # Update and log the ema value
        # TODO : make the ema param configurable
        if self.avg_reward_against_minimax_ema is None:
            self.avg_reward_against_minimax_ema = avg_reward_against_minimax
        else:
            self.avg_reward_against_minimax_ema = ((0.85 * self.avg_reward_against_minimax_ema)
                                                   + (0.15 * avg_reward_against_minimax))

        self.log('avg_reward_against_minimax_ema', self.avg_reward_against_minimax_ema)


def main():
    # Start wandb run
    wandb_kwargs = dict(
        project='aigames2',
        group='tictactoe_lightning',
        reinit=False,
    )

    # So we can get the name for model checkpointing
    wandb.init(**wandb_kwargs)
    wandb_run = wandb.run.name or os.path.split(wandb.run.path)[-1]

    network = TicTacToeNetwork()
    evaluator = FastTicTacToeEvaluator(network)
    hyperparams = AlphaTrainingHyperparametersLightning()
    hyperparams.min_data_size = 256
    hyperparams.n_mcts = 100
    hyperparams.dirichlet_alpha = 0.3
    hyperparams.dirichlet_epsilon = 0.5
    hyperparams.lr = 0.001
    hyperparams.weight_decay = 1e-4
    hyperparams.training_tau = TrainingTau(0.5)
    hyperparams.c_puct = 5
    hyperparams.batch_size = 64

    # Setup an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default=f'./ckpt/')
    for slot in get_all_slots(hyperparams):
        parser.add_argument(f'--{slot}', type=type(getattr(hyperparams, slot)), default=getattr(hyperparams, slot))

    # Parse the args and set the hyperparams
    args = parser.parse_args()
    for slot in get_all_slots(hyperparams):
        setattr(hyperparams, slot, getattr(args, slot))

    training_run = AlphaTrainingRunLightningTTT(FastTicTacToe, evaluator, hyperparams)
    model_checkpoint = ModelCheckpoint(dirpath=os.path.join(args.ckpt_dir, wandb_run), save_top_k=1, mode='max',
                         monitor='avg_reward_against_minimax_ema')
    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=1, logger=pl.loggers.WandbLogger(**wandb_kwargs),
                         callbacks=[model_checkpoint])
    trainer.fit(training_run)


if __name__ == '__main__':
    main()
