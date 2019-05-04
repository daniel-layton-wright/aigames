import os
import sys
top_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))
sys.path.insert(0, top_dir)
from aigames import *
import torch.multiprocessing as mp
import logging
import wandb
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class TicTacToeAlphaModel(AlphaModel):
    def __init__(self, game_class):
        super().__init__()
        self.game_class = game_class
        self.base = nn.Sequential(
            nn.Conv2d(in_channels = game_class.STATE_SHAPE[0], out_channels = 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.base_out_features = 32 * game_class.STATE_SHAPE[1] * game_class.STATE_SHAPE[2]

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=len(game_class.ALL_ACTIONS)),
            nn.Softmax(dim=0)
        )
        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=1),
            nn.Tanh()
        )

    def forward(self, state):
        state_processed = self.process_state(state)
        base = self.base(state_processed)
        policy = self.policy_head(base).squeeze()
        value = self.value_head(base).squeeze()
        return policy, value

    @staticmethod
    def process_state(state):
        x = torch.FloatTensor(state)
        while len(x.shape) < 4:
            x = x.unsqueeze(0)

        return x


class Monitor(AlphaMonitor):
    def __init__(self):
        self.train_iter_count = mp.Value('i', 0)

    def before_training_start(self):
        wandb.init(project='aigames', tags='tictactoe_alpha')

    def on_optimizer_step(self, most_recent_loss):
        with self.train_iter_count.get_lock():
            self.train_iter_count.value += 1
            wandb.log({'iter': self.train_iter_count.value, 'loss': most_recent_loss})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--n_self_play_procs', type=int, default=1, help='Number of self-play processes')
    parser.add_argument('-n', '--n_games_per_proc', type=int, required=True, help='Number of self-play games per process')
    parser.add_argument('--n_training_workers', type=int, default=1, help='Number of self-play games per process')
    args = parser.parse_args()

    mp.set_start_method('forkserver')

    model = TicTacToeAlphaModel(TicTacToe)
    monitor = Monitor()
    train_alpha_agent_mp(TicTacToe, model, monitor=monitor, n_self_play_procs=args.n_self_play_procs,
                         n_games_per_proc=args.n_games_per_proc, n_training_workers=args.n_training_workers)


if __name__ == '__main__':
    main()
