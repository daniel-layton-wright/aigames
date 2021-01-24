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
import matplotlib.pyplot as plt


class TicTacToeTrainingListenerAlpha(TicTacToeTrainingListener):
    def __init__(self):
        self.training_manager = None
        self.minimax_agent = MinimaxAgent(FastTicTacToe)
        self.test_state = FastTicTacToeState(np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [-1., 0., 0.]
        ]))

    def before_begin_training(self, training_manager: AlphaTrainingRun):
        wandb.init(project='aigames2', tags=[training_manager.game.__name__, 'alpha'], tensorboard=True)
        params = {key: val for key, val in training_manager.__dict__.items() if is_json_serializable(val)}
        wandb.config.update(params)

        self.training_manager = training_manager

    def on_training_step(self, iter: int, loss: float, training_manager, **kwargs):
        policy, value = training_manager.alpha_evaluators[-1].evaluate(self.test_state)
        data = [[label, val] for (label, val) in zip(list(range(len(policy))), policy)]
        table = wandb.Table(data=data, columns=["label", "value"])
        log_dict = {'iter': iter, 'loss': loss, 'test_state_policy': wandb.plot.bar(table, "label", "value", title="Test State Policy")}

        if iter % 750 == 0:
            log_dict['avg_reward_against_minimax'] = self.play_tournament_against_minimax(100)

        wandb.log(log_dict)

    def play_tournament_against_minimax(self, n_games):
        agent = self.training_manager.agents[-1]
        agent.eval()
        discount_rate = agent.discount_rate
        reward_listener = RewardListener(discount_rate, 1)

        rewards = []
        for _ in range(n_games):
            game = FastTicTacToe([self.minimax_agent, agent], [reward_listener])
            game.play()
            rewards.append( reward_listener.reward )

        agent.train()

        return float(sum(rewards)) / len(rewards)


class TicTacToeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            Flatten()
        )

        self.base_out_features = 16 * 3 * 3

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=9),
            nn.Softmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=1),
            nn.Tanh()
        )

    def forward(self, processed_state):
        base = self.base(processed_state)
        policy = self.policy_head(base).squeeze()
        value = self.value_head(base).squeeze()
        return policy, value


class TicTacToeEvaluator(AlphaNetworkEvaluator):
    def process_state(self, state):
        if abs(state).sum() % 1 == 1:
            state = copy.deepcopy(state)
            state *= -1
        return torch.FloatTensor(state).reshape(1, 3, 3)


class FastTicTacToeEvaluator(AlphaNetworkEvaluator):
    def process_state(self, state: FastTicTacToeState):
        s = state.tensor_state
        if abs(s).sum() % 1 == 1:
            s = copy.deepcopy(s)
            s *= -1
        return s


def main():
    network = TicTacToeNetwork()
    evaluator = FastTicTacToeEvaluator(network)
    training_listener = TicTacToeTrainingListenerAlpha()
    training_run = AlphaTrainingRun(FastTicTacToe, evaluator, lambda params: torch.optim.Adam(params, lr=0.005, weight_decay=1e-5), TrainingTau(tau_schedule_function=lambda x: 0.5**x),
                                    min_data_size=100, dirichlet_alpha=0.03, dirichlet_epsilon=0.25,
                                    training_listeners=[training_listener])
    training_run.train(n_games=3000)


if __name__ == '__main__':
    main()
