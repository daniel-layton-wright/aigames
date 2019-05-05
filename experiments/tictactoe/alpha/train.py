import os
import sys

top_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))
sys.path.insert(0, top_dir)
from aigames import *
import torch.multiprocessing as mp
import logging
import wandb
from tensorboardX import SummaryWriter

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class TicTacToeAlphaModel(AlphaModel):
    def __init__(self, game_class):
        super().__init__()
        self.game_class = game_class
        self.base = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.base_out_features = 32 * game_class.STATE_SHAPE[1] * game_class.STATE_SHAPE[2]

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=len(game_class.ALL_ACTIONS)),
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

    @staticmethod
    def process_state(state):
        # Make the state from the perspective of the player whose turn it is
        if state[2, 0, 0] == 0:
            x = torch.FloatTensor(state[[0, 1]])
        else:
            x = torch.FloatTensor(state[[1, 0]])

        # Make dimensions correct
        while len(x.shape) < 4:
            x = x.unsqueeze(0)

        return x


class Monitor(MultiprocessingAlphaMonitor):
    def __init__(self, model, agent, kill, pause_training, train_iter_queue: mp.Queue, args, model_device,
                 evaluate_every_n_iters: int = 1):
        self.model = model
        self.agent = agent
        self.tournament = Tournament(self.model.game_class, [MinimaxAgent(self.model.game_class), self.agent],
                                     n_games=10)
        self.train_iter_count = mp.Value('i', 0)
        self.train_iter_queue = train_iter_queue
        self.args = args
        self.kill = kill
        self.logger = None
        self.model_device = torch.device(model_device)
        self.pause_training = pause_training
        self.evaluate_every_n_iters = evaluate_every_n_iters

        example_state = np.zeros((3, 3, 3))
        example_state[0, 1, 1] = 1
        example_state[1, 0, 1] = 1
        example_state[2, :, :] = 0
        self.processed_example_state = self.model.process_state(example_state).to(self.model_device)

    def start(self):
        wandb.init(project='aigames', tags=['tictactoe_alpha'], tensorboard=True)
        wandb.config.update(self.args)
        self.logger = SummaryWriter()

    def on_optimizer_step(self, most_recent_loss):
        print('on_optimizer_step')
        with self.train_iter_count.get_lock():
            self.train_iter_count.value += 1
            self.train_iter_queue.put({'iter': self.train_iter_count.value, 'loss': most_recent_loss})

    def monitor_until_killed(self):
        do_one_last_check = True
        while True:
            try:
                cur_log_dict = self.train_iter_queue.get(block=False)
            except queue.Empty:
                if self.kill.value:
                    if do_one_last_check:
                        time.sleep(1)
                        do_one_last_check = False
                        continue
                    else:
                        break
                else:
                    continue

            wandb.log(cur_log_dict)

            if self.train_iter_count.value % self.evaluate_every_n_iters == 0:
                self.pause_training.value = True
                self.agent.eval()

                with torch.no_grad():
                    p, v = self.model(self.processed_example_state)

                p = p.cpu().numpy()
                fig, ax = plt.subplots()
                ax.bar(range(len(p)), p)
                ax.set_xticks(range(len(p)))
                ax.set_xticklabels(self.model.game_class.ALL_ACTIONS)
                self.logger.add_figure('test_state_actions', fig, self.train_iter_count.value)

                wandb.log({'iter': cur_log_dict['iter'],
                           'example_state_actions': fig})
                plt.close(fig)

                final_states = self.tournament.play()
                num_losses = sum(self.model.game_class.reward(state, 1) == -1 for state in final_states)
                pct_losses = num_losses / float(self.tournament.n_games)

                wandb.log({'iter': cur_log_dict['iter'],
                           'pct_loss_vs_minimax': pct_losses})

                self.pause_training.value = False
                self.agent.train()


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lr', type=float, default=0.005)
    parser.add_argument('-p', '--n_self_play_procs', type=int, default=1, help='Number of self-play processes')
    parser.add_argument('-n', '--n_games_per_proc', type=int, required=True,
                        help='Number of self-play games per process')
    parser.add_argument('--n_training_workers', type=int, default=1, help='Number of self-play games per process')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--optimizer_class', type=str, default='Adam', help='optimizer class in torch.optim')
    parser.add_argument('--evaluate_every_n_iters', type=int, default=1000,
                        help='how often to play tournament against minimax')
    return parser


def run(args, model, monitor=None):
    optimizer_class = eval(f'torch.optim.{args.optimizer_class}')
    train_alpha_agent_mp(TicTacToe, model, optimizer_class=optimizer_class, lr=args.lr, model_device=args.device,
                         monitor=monitor, n_self_play_workers=args.n_self_play_procs,
                         n_games_per_worker=args.n_games_per_proc, n_training_workers=args.n_training_workers)


def main():
    parser = get_parser()
    args = parser.parse_args()

    mp.set_start_method('forkserver')
    model = TicTacToeAlphaModel(TicTacToe)
    monitor_log_queue = mp.Queue(maxsize=1000)
    monitor = lambda model, agent, kill, pause_training: Monitor(model, agent, kill, pause_training,
                                                                 monitor_log_queue, args, args.device,
                                                                 evaluate_every_n_iters=args.evaluate_every_n_iters)

    run(args, model, monitor)


if __name__ == '__main__':
    main()
