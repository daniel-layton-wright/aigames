from aigames import *
from aigames.game.tictactoe import *
import torch.optim
import wandb
import torch.multiprocessing as mp


class TicTacToeTrainingListener(TrainingListener):
    def __init__(self):
        self.training_manager = None
        self.minimax_agent = None

    def before_begin_training(self, training_manager: MPQLearningTrainingRun):
        wandb.init(project='aigames2', tags=[training_manager.game.__name__, 'qlearning'], tensorboard=True)
        params = {key: val for key, val in training_manager.__dict__.items() if is_json_serializable(val)}
        wandb.config.update(params)
        self.training_manager = training_manager

    def on_training_step(self, iter: int, loss: float, training_manager: MPQLearningTrainingRun, **kwargs):
        n = kwargs.get('network_num', 0) # network num
        log_dict = {f'iter_{n}': iter, f'loss_{n}': loss}
        log_dict[f'terminal_loss_{n}'] = kwargs.get('terminal_loss', None)
        log_dict[f'nonterminal_loss_{n}'] = kwargs.get('nonterminal_loss', None)
        log_dict[f'exploration_probability_{n}'] = training_manager.exploration_probability_schedulers[n].get_exploration_probability(None, None)

        if iter % 100 == 0:
            if self.training_manager.share_among_players:
                log_dict[f'avg_reward_against_minimax_0'] = self.play_tournament_against_minimax(100, n, 0)
                log_dict[f'avg_reward_against_minimax_1'] = self.play_tournament_against_minimax(100, n, 1)
            else:
                log_dict[f'avg_reward_against_minimax_{n}'] = self.play_tournament_against_minimax(100, n, n)

        wandb.log(log_dict)

    def play_tournament_against_minimax(self, n_games, network_num, play_as_agent_num):
        network = copy.deepcopy(self.training_manager.networks[network_num])
        q_function = BasicQLearningFunction(network, self.training_manager.state_shape)
        agent = QLearningAgent(self.training_manager.game, play_as_agent_num, q_function, data_listener=None,
                               exploration_probability_scheduler=None, discount_rate=self.training_manager.discount_rate)
        agent.eval()
        discount_rate = agent.discount_rate
        reward_listener = RewardListener(discount_rate, play_as_agent_num)

        rewards = []
        players = [None, None]
        players[play_as_agent_num] = agent
        players[(1 - play_as_agent_num)] = MinimaxAgent(self.training_manager.game)
        game = self.training_manager.game(players, [reward_listener])
        for _ in range(n_games):
            game.play()
            rewards.append( reward_listener.reward )

        agent.train()

        return float(sum(rewards)) / len(rewards)


class RewardListener(GameListener):
    def __init__(self, discount_rate, player_index):
        self.reward = 0
        self.i = 0
        self.discount_rate = discount_rate
        self.player_index = player_index

    def before_game_start(self, game):
        self.reward = 0
        self.i = 0

    def on_action(self, game, action):
        next_state, rewards = game.get_next_state_and_rewards(game.state, action)
        self.reward += (self.discount_rate ** self.i) * rewards[self.player_index]
        self.i += 1


def train_tictactoe_v1():
    network = lambda: nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=64 * 3 * 3, out_features=9),
        nn.Tanh()
    )

    optimizer = lambda net: torch.optim.Adam(net.parameters(), lr=1e-3)
    training_listener = TicTacToeTrainingListener()
    training_manager = MPQLearningTrainingRun(TicTacToe, network, optimizer, state_shape=(1, 3, 3),
                                              training_listeners=[training_listener],
                                              update_target_network_ever_n_iters=200,
                                              n_self_play_workers=4, n_training_workers=1,
                                              min_data_size=500, max_data_size=10000, batch_size=32,
                                              frac_terminal_to_sample=0.75
                                              )
    training_manager.train(n_games=20000)


def train_tictactoe_v2():
    network = lambda: nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=64 * 3 * 3, out_features=9),
        nn.Tanh()
    )

    optimizer = lambda net: torch.optim.Adam(net.parameters(), lr=1e-3)
    training_listener = TicTacToeTrainingListener()
    training_manager = MPQLearningTrainingRun(TicTacToe, network, optimizer, state_shape=(1, 3, 3),
                                              training_listeners=[training_listener], share_among_players=False,
                                              update_target_network_ever_n_iters=200,
                                              n_self_play_workers=4, n_training_workers=1,
                                              exploration_probability_half_life=5000,
                                              min_data_size=500, max_data_size=10000, batch_size=32,
                                              frac_terminal_to_sample=0.5, data_debug_mode=False
                                              )
    training_manager.train(n_games=60000)
    import pdb; pdb.set_trace()
    # training_listener.play_tournament_against_minimax(10)


def main():
    mp.set_start_method('forkserver')
    train_tictactoe_v2()


if __name__ == '__main__':
    main()
