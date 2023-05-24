import sys

sys.path.append('/Users/dlwright/Documents/Projects/aigames/')

from aigames.game.tictactoe import TicTacToe
from aigames.agent.alpha_agent import *
from aigames.agent.minimax_agent import MinimaxAgent
from aigames.game.command_line_game import CommandLineGame
from aigames.utils.utils import play_tournament
from aigames.training_manager.alpha_training_manager import *
from aigames.utils.listeners import RewardListener
from aigames.game.tictactoe import *
from aigames import Flatten
import torch
from qlearning_tictactoe import TicTacToeTrainingListener
import wandb
import wandb.plot
import optuna
import typing
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger


class TicTacToeTrainingListenerAlpha(TicTacToeTrainingListener):
    def __init__(self, optuna_trial: typing.Union[None, optuna.Trial] = None):
        super().__init__("aigames2", ['tictactoe', 'alpha'])
        self.training_manager = None
        self.minimax_agent = MinimaxAgent(FastTicTacToe)
        self.test_state = FastTicTacToeState(np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [-1., 0., 0.]
        ]))

        self.get_game_name = lambda training_manager: training_manager.game.__name__
        self.get_alpha_evaluators = lambda training_manager: training_manager.alpha_evaluators
        self.ema_reward_minimax = None
        self.optuna_trial = optuna_trial

    def before_begin_training(self, training_manager: AlphaTrainingRun):
        if wandb.run:
            wandb.run.finish()

        wandb.init(project='aigames2', tags=[self.get_game_name(training_manager), 'alpha'])
        params = {key: val for key, val in training_manager.__dict__.items() if is_json_serializable(val)}
        wandb.config.update(params)

        self.ema_reward_minimax = None

        for i, evaluator in enumerate(self.get_alpha_evaluators(training_manager)):
            artifact = wandb.Artifact(f'model{i}', type='model')
            f = f'{wandb.run.dir}/model{i}.pt'
            torch.onnx.export(evaluator.network, evaluator.process_state(FastTicTacToeState(TicTacToe.get_initial_state())).unsqueeze(0), f)
            artifact.add_file(f)
            wandb.run.log_artifact(artifact)
            wandb.watch(evaluator.network, log="all")

        self.training_manager = training_manager

    def on_training_step(self, iter: int, loss: float, training_manager, **kwargs):
        policy, value = self.get_alpha_evaluators(training_manager)[-1].evaluate(self.test_state)
        data = [[label, val] for (label, val) in zip(list(range(len(policy))), policy)]
        table = wandb.Table(data=data, columns=["label", "value"])
        log_dict = {'iter': iter, 'loss': loss}

        if iter % 50 == 0:
            log_dict['avg_reward_against_minimax'] = self.play_tournament_against_minimax(100)

            if self.ema_reward_minimax is None:
                self.ema_reward_minimax = log_dict['avg_reward_against_minimax']

            self.ema_reward_minimax = (0.85 * self.ema_reward_minimax) + (0.15 * log_dict['avg_reward_against_minimax'])

            if self.optuna_trial:
                self.optuna_trial.report(self.ema_reward_minimax, iter)
                if self.optuna_trial.should_prune():
                    raise optuna.TrialPruned()

        wandb.log(log_dict)

        if iter % 50 == 0:
            wandb.log({'test_state_policy': wandb.plot.bar(table, "label", "value", title="Test State Policy")})

    def play_tournament_against_minimax(self, n_games):
        agent = self.training_manager.agents[-1]
        agent.eval()
        discount_rate = agent.hyperparams.discount_rate
        reward_listener = RewardListener(discount_rate, 1)

        rewards = []
        for _ in range(n_games):
            game = FastTicTacToe([self.minimax_agent, agent], [reward_listener])
            game.play()
            rewards.append( reward_listener.reward )

        agent.train()

        return float(sum(rewards)) / len(rewards)


class TicTacToeTrainingListenerAlphaLightning(TicTacToeTrainingListenerAlpha):
    def __init__(self):
        super().__init__()

        self.get_game_name = lambda training_manager: training_manager.game_class.__name__
        self.get_alpha_evaluators = lambda training_manager: [training_manager.alpha_evaluator]


class TicTacToeTrainingCallback(pl.Callback):
    def __init__(self, training_listener):
        super().__init__()
        self.training_listener = training_listener
        self.i = 0
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.i += 1
        print(f'batch idx: {batch_idx}, batch shape: {batch[0].shape}')
        self.training_listener.on_training_step(self.i, outputs['loss'], pl_module)


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


class TicTacToeEvaluator(AlphaNetworkEvaluator):
    def process_state(self, state):
        if abs(state).sum() % 2 == 1:
            state = copy.deepcopy(state)
            state *= -1
        return torch.FloatTensor(state).reshape(1, 3, 3)


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
    def objective(trial: optuna.Trial):
        network = TicTacToeNetwork()
        evaluator = FastTicTacToeEvaluator(network)
        training_listener = TicTacToeTrainingListenerAlpha(trial)
        hyperparams = AlphaTrainingHyperparameters()
        hyperparams.min_data_size = 10
        hyperparams.n_mcts = 100
        hyperparams.dirichlet_alpha = trial.suggest_float('dirichlet_alpha', 0.01, 0.5, log=True)
        hyperparams.dirichlet_epsilon = trial.suggest_float('dirichlet_epsilon', 0.2, 0.8)
        hyperparams.lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        hyperparams.weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
        hyperparams.training_tau = TrainingTau(trial.suggest_float('training_tau', 0.3, 3, log=True))
        hyperparams.c_puct = trial.suggest_float('c_puct', 0.1, 10, log=True)
        hyperparams.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        training_run = AlphaTrainingRun(FastTicTacToe, evaluator, lambda params: torch.optim.Adam(params, lr=hyperparams.lr,
                                                                                                  weight_decay=hyperparams.weight_decay),
                                        hyperparams,
                                        training_listeners=[training_listener])

        training_run.train(n_games=1000)
        return training_listener.ema_reward_minimax

    from optuna.storages import JournalFileStorage, JournalStorage
    storage = JournalStorage(JournalFileStorage(f'{os.getcwd()}/optuna_experiment.log'))
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), storage=storage)
    study.optimize(objective, n_trials=60)

    network = TicTacToeNetwork()
    evaluator = FastTicTacToeEvaluator(network)
    training_listener = TicTacToeTrainingListenerAlphaLightning()
    callback = TicTacToeTrainingCallback(training_listener)
    hyperparams = AlphaTrainingHyperparametersLightning()
    hyperparams.min_data_size     = 256
    hyperparams.n_mcts            = 100
    hyperparams.dirichlet_alpha   = 0.3
    hyperparams.dirichlet_epsilon = 0.5
    hyperparams.lr                = 0.001
    hyperparams.weight_decay      = 1e-4
    hyperparams.training_tau      = TrainingTau(0.5)
    hyperparams.c_puct            = 5
    hyperparams.batch_size        = 64

    # training_run = AlphaTrainingRunLightning(FastTicTacToe, evaluator, hyperparams,
    #                                          training_listeners=[training_listener])
    #
    # trainer = pl.Trainer(reload_dataloaders_every_n_epochs=1, logger=WandbLogger(), callbacks=[callback])
    # trainer.fit(training_run)

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    main()
