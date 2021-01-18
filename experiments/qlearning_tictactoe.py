from aigames import *
from aigames.game.tictactoe import *
from aigames.utils.listeners import RewardListener
import torch.optim
import wandb


class TicTacToeTrainingListener(TrainingListener):
    def __init__(self):
        self.training_manager = None
        self.minimax_agent = MinimaxAgent(TicTacToe)

    def before_begin_training(self, training_manager):
        wandb.init(project='aigames2', tags=[training_manager.game.__name__, 'qlearning'], tensorboard=True)
        params = {key: val for key, val in training_manager.__dict__.items() if is_json_serializable(val)}
        wandb.config.update(params)

        self.training_manager = training_manager

    def on_training_step(self, iter: int, loss: float, training_manager, **kwargs):

        test_state = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [-1., 0., 0.]
        ])
        scores = training_manager.q_functions[-1].evaluate(test_state)

        log_dict = {'iter': iter, 'loss': loss, 'test_score8': scores[8], 'test_score7': scores[7]}

        if iter % 100 == 0:
            log_dict['avg_reward_against_minimax'] = self.play_tournament_against_minimax(100)

        wandb.log(log_dict)

    def play_tournament_against_minimax(self, n_games):
        agent = self.training_manager.agents[-1]
        agent.eval()
        discount_rate = agent.discount_rate
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
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=9),
        nn.Tanh()
    )

    optimizer = lambda net: torch.optim.Adam(net.parameters(), lr=5e-3)
    training_listener = TicTacToeTrainingListener()
    training_manager = QLearningTrainingRun(TicTacToe, network4, optimizer, state_shape=(1, 3, 3), training_listeners=[training_listener],
                                            update_target_network_ever_n_iters=500, batch_size=128,
                                            share_among_players=False,
                                            frac_terminal_to_sample=0.8,
                                            exploration_probability=0.2
                                            )
    training_manager.train(n_games=50000)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
