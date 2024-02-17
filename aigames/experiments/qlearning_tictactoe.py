from aigames import *
from aigames.game.tictactoe import *
from aigames.utils.listeners import RewardListener
import torch.optim
import wandb
import typing


class TicTacToeTrainingListener(TrainingListener):
    def __init__(self, project, tags):
        self.training_manager : typing.Union[None, QLearningTrainingRun] = None
        self.minimax_agent = MinimaxAgent(TicTacToe)
        self.project = project
        self.tags = tags

    def before_begin_training(self, training_manager: QLearningTrainingRun):
        wandb.init(project=self.project, tags=self.tags, tensorboard=True)
        from itertools import chain
        slots = chain.from_iterable(getattr(cls, '__slots__', []) for cls in type(training_manager.hyperparams).__mro__)
        params = {s: getattr(training_manager.hyperparams, s) for s in slots if hasattr(training_manager.hyperparams, s)}
        wandb.config.update(params)

        for i, network in enumerate(training_manager.networks):
            artifact = wandb.Artifact(f'model{i}', type='model')
            f = f'{wandb.run.dir}/model{0}.pt'
            torch.onnx.export(network, torch.FloatTensor(np.random.randn(*training_manager.hyperparams.state_shape)), f)
            artifact.add_file(f)
            wandb.run.log_artifact(artifact)
            wandb.watch(network, log="all")

        self.training_manager = training_manager

    def on_training_step(self, iter: int, loss: float, training_manager: QLearningTrainingRun, **kwargs):

        test_state = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [-1., 0., 0.]
        ])
        scores = training_manager.q_functions[-1].evaluate( test_state)

        log_dict = {'iter': iter, 'loss': loss, 'test_score8': scores[8], 'test_score7': scores[7],
                    # hard coded but whatever:
                    'exploration_prob': training_manager.hyperparams.exploration_probability,
                    'lr': training_manager.optimizers[0].param_groups[0]['lr'],
                    'weight_decay': training_manager.optimizers[0].param_groups[0]['weight_decay']}

        if iter % 100 == 0:
            log_dict['avg_reward_against_minimax'] = self.play_tournament_against_minimax(100)

        wandb.log(log_dict)

    def play_tournament_against_minimax(self, n_games):
        agent = self.training_manager.agents[-1]
        agent.eval()
        discount_rate = self.training_manager.hyperparams.discount_rate
        reward_listener = RewardListener(discount_rate, 1)

        rewards = []
        for _ in range(n_games):
            game = TicTacToe([self.minimax_agent, agent], [reward_listener])
            game.play()
            rewards.append( reward_listener.reward )

        agent.train()

        return float(sum(rewards)) / len(rewards)

    def play_minimax_debug_game(self):
        cli = CommandLineGame(clear_screen=False)
        debugger = DebugGameListener()
        self.training_manager.agents[-1].eval()
        game = TicTacToe([self.minimax_agent, self.training_manager.agents[-1]], listeners=[cli, debugger])
        game.play()


class StepExplorationProbabilityScheduler(TrainingListener):
    def __init__(self, frequency, step_size, min=0.0):
        self.frequency = frequency
        self.step_size = step_size
        self.min = min

    def on_training_step(self, iter: int, loss: float, training_manager, **kwargs):
        if (iter % self.frequency) == 0 and iter > 0:
            new_exp = max(training_manager.hyperparams.exploration_probability - self.step_size, self.min)
            training_manager.hyperparams.exploration_probability = new_exp


class QLearningHyperparameters(QLearningTrainingHyperparameters):
    __slots__ = ['win_reward', 'tie_reward', 'lose_reward']

    def __init__(self):
        super().__init__()
        self.win_reward = 1
        self.tie_reward = 0
        self.lose_reward = -1


def main():
    network = lambda: nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=64 * 3 * 3, out_features=9),
        nn.Tanh()
    )

    network2 = lambda: nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=64 * 3 * 3, out_features=9),
        nn.Tanh()
    )

    network3 = lambda: nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=8 * 3 * 3, out_features=9),
        nn.Tanh()
    )

    network4 = lambda: nn.Sequential(
        Flatten(),
        nn.Linear(in_features=9, out_features=64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=9),
        nn.BatchNorm1d(9),
        nn.Tanh()
    )

    network5 = lambda: nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=16 * 3 * 3, out_features=9),
        nn.Tanh()
    )

    class Network6(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = Flatten()
            self.lin1 = nn.Linear(9, 32)
            self.lin2 = nn.Linear(32, 32)
            self.lin3 = nn.Linear(32, 3*9)

        def forward(self, x):
            x = self.flatten(x)
            x = nn.functional.relu(self.lin1(x))
            x = nn.functional.dropout(x, 0.1, self.training)
            x = x + nn.functional.dropout(nn.functional.relu(self.lin2(x)), 0.1, self.training)
            x = self.lin3(x)
            x = x.reshape(x.shape[0], 9, 3)
            x = nn.functional.softmax(x, 2)
            x = x @ torch.FloatTensor([-1, 0, 1])
            return x


    class Network7(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = Flatten()
            self.lin1 = nn.Linear(9, 32)
            self.lin2 = nn.Linear(32, 32)
            self.lin3 = nn.Linear(32, 9)

        def forward(self, x):
            x = self.flatten(x)
            x = nn.functional.relu(self.lin1(x))
            x = nn.functional.dropout(x, 0.1, self.training)
            x = x + nn.functional.dropout(nn.functional.relu(self.lin2(x)), 0.1, self.training)
            x = self.lin3(x)
            return x

    optimizer = lambda net: torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4)
    training_listener = TicTacToeTrainingListener('aigames2', ['tictactoe', 'qlearning'])
    #exp_prob_sched = StepExplorationProbabilityScheduler(25000, 0.05, min=0.02)
    train_eval_alternator = TrainEvalAlternator()

    hyperparams = QLearningHyperparameters()
    hyperparams.update_target_network_every_n_iters = 3000
    hyperparams.batch_size = 32
    hyperparams.share_among_players = False
    hyperparams.frac_terminal_to_sample = None
    hyperparams.exploration_probability = 0.7
    hyperparams.max_data_size = 20000
    hyperparams.use_minimax_agent0 = False
    hyperparams.win_reward = 1
    hyperparams.tie_reward = 0
    hyperparams.lose_reward = -1

    game_class = custom_reward_tictactoe(hyperparams.win_reward, hyperparams.tie_reward, hyperparams.lose_reward)
    training_manager = QLearningTrainingRun(game_class, Network7, optimizer, hyperparams,
                                            training_listeners=[training_listener,
                                                                #exp_prob_sched,
                                                                train_eval_alternator
                                                                ])
    training_manager.train(n_games=1000)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
