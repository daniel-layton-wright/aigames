from aigames import *
from aigames.game.twenty_forty_eight import *
import torch.optim
import wandb


class TwentyFourtyEightTrainingListener(TrainingListener):
    def __init__(self):
        self.training_manager = None
        self.cum_reward = 0
        self.move_num = 0
        self.discount_rate = None
        self.player_index = 0
        self.iter = 0

    def before_game_start(self, game):
        self.cum_reward = 0
        self.move_num = 0

    def on_action(self, game, action):
        _, rewards = game.get_next_state_and_rewards(game.state, action)
        self.cum_reward += (self.discount_rate ** self.move_num) * rewards[self.player_index]
        self.move_num += 1

    def on_game_end(self, game):
        wandb.log({'iter': self.iter, 'reward': self.cum_reward})

    def before_begin_training(self, training_manager):
        wandb.init(project='aigames2', tags=[training_manager.game.__name__, 'qlearning'], tensorboard=True)
        params = {key: val for key, val in training_manager.__dict__.items() if is_json_serializable(val)}
        wandb.config.update(params)

        self.training_manager = training_manager
        self.discount_rate = self.training_manager.agents[0].discount_rate

    def on_training_step(self, iter: int, loss: float, training_manager):
        self.iter = iter
        log_dict = {'iter': iter, 'loss': loss}
        wandb.log(log_dict)


class Exponentiate(nn.Module):
    def forward(self, input):
        return torch.exp(input)


def main():
    network = lambda: nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=32 * 4 * 4, out_features=8),
        nn.ReLU(),
        nn.Linear(in_features=8, out_features=4)
    )
    optimizer = lambda network: torch.optim.Adam(network.parameters())
    training_listener = TwentyFourtyEightTrainingListener()
    training_manager = QLearningTrainingRun(TwentyFortyEight, network, optimizer, state_shape=(1, 4, 4),
                                            training_listeners=[training_listener], discount_rate=0.99,
                                            update_target_network_ever_n_iters=1000)
    training_manager.train(n_games=5000)

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    main()
