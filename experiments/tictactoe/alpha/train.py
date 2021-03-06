import os
import sys
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

top_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))
sys.path.insert(0, top_dir)
from aigames import *
from aigames.train_utils.train_alpha_agent import *
from aigames.train_utils.train_alpha_agent_mp import *
import torch.multiprocessing as mp
import logging
import wandb
from tensorboardX import SummaryWriter
from threading import Thread

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class TicTacToeAlphaModel(AlphaModel):
    def __init__(self, optimizer_creator=None, device=torch.device('cpu'), monitor=None):
        super().__init__(device, monitor)
        self.base = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
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

        self.setup(optimizer_creator)

    def forward(self, processed_state):
        base = self.base(processed_state)
        policy = self.policy_head(base).squeeze()
        value = self.value_head(base).squeeze()
        return policy, value

    def process_state(self, state):
        # Make the state from the perspective of the player whose turn it is
        if state[2, 0, 0] == 0:
            x = torch.FloatTensor(state[[0, 1]])
        else:
            x = torch.FloatTensor(state[[1, 0]])

        # Make dimensions correct
        while len(x.shape) < 4:
            x = x.unsqueeze(0)

        return x.to(self.device)


class Monitor(MultiprocessingAlphaMonitor):
    def __init__(self, model, agent, args, train_iter_queue: mp.Queue = None, kill=None,
                 pause_training=None, evaluate_every_n_iters: int = 1, n_games_in_evaluation: int = 100,
                 wandb: bool = True):
        self.model = model
        self.agent = agent
        self.train_iter_count = mp.Value('i', 0)
        self.train_iter_queue = train_iter_queue
        self.n_games_in_evaluation = n_games_in_evaluation
        self.args = args
        self.kill = kill
        self.logger = None
        self.pause_training = pause_training
        self.evaluate_every_n_iters = evaluate_every_n_iters
        self.best_frac_losses = None
        self.wandb = wandb
        self.thread_pool = None
        self.cur_model = None
        self.evaluation_thread = None
        self.currently_evaluating = False
        self.evaluation_results = None
        self.evaluation_results_ready = False

        example_state = np.zeros((3, 3, 3))
        example_state[0, 1, 1] = 1
        example_state[1, 0, 1] = 1
        example_state[2, :, :] = 0
        self.processed_example_state = self.model.process_state(example_state)

        example_state2 = np.array([[[0, 0, 0],
                                    [1, 0, 0],
                                    [1, 0, 1]],
                                   [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]],
                                   [[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]]]).astype(int)
        self.processed_example_state2 = self.model.process_state(example_state2)

    def start(self):
        if self.wandb:
            wandb.init(project='aigames', tags=['tictactoe_alpha'], tensorboard=True)
            wandb.config.update(self.args)
        self.logger = SummaryWriter()

    def on_optimizer_step(self, most_recent_loss):
        with self.train_iter_count.get_lock():
            self.train_iter_count.value += 1

            cur_log_item = {'iter': self.train_iter_count.value, 'loss': most_recent_loss}
            if self.train_iter_queue is not None:
                self.train_iter_queue.put(cur_log_item)
            else:
                self.log(cur_log_item)

    def monitor_until_killed(self):
        do_one_last_check = True

        while True:
            self.log_evaluation_results_if_available()
            try:
                cur_log_dict = self.train_iter_queue.get(block=False)
            except queue.Empty:
                if self.kill.value:
                    self.wait_for_and_log_evaluation_results()
                    if do_one_last_check:
                        time.sleep(1)
                        do_one_last_check = False
                        continue
                    else:
                        break
                else:
                    continue

            self.log(cur_log_dict)

    def log(self, cur_log_dict):
        self._log(cur_log_dict)

        self.log_action_prob_plot(self.processed_example_state, 'example_state_actions')
        self.log_action_prob_plot(self.processed_example_state2, 'example_state_actions2')

        if cur_log_dict['iter'] % self.evaluate_every_n_iters == 0:
            # Pause while we make a copy
            if self.pause_training is not None:
                self.pause_training.value = True

            # Make a copy of the agent to evaluate with
            iter = self.train_iter_count.value  # note the actual iteration step that is being evaluated
            self.cur_model = copy.copy(self.model)
            self.cur_model.load_state_dict(self.model.state_dict())
            cur_agent: AlphaAgent = copy.copy(self.agent)
            cur_agent.evaluator = self.cur_model  # Don't do multiprocessing here, just use the model
            cur_agent.eval()

            # Unpause
            if self.pause_training is not None:
                self.pause_training.value = False

            if not self.args.single_process:
                self.wait_for_and_log_evaluation_results()
                self.currently_evaluating = True
                self.evaluation_results_ready = False
                self.evaluation_thread = Thread(target=self.play_minimax_tournament_with_current_model,
                                                args=(cur_agent, iter,
                                                      self.n_games_in_evaluation))
                self.evaluation_thread.start()
            else:
                results = self.play_minimax_tournament_with_current_model(cur_agent, iter,
                                                                          self.n_games_in_evaluation)
                self.log_evaluation_results(results)

    def play_minimax_tournament_with_current_model(self, agent, iter, n_games_in_evaluation):
        tournament = Tournament(agent.game_class, [MinimaxAgent(agent.game_class), agent],
                                n_games=n_games_in_evaluation)
        final_states = tournament.play()
        num_losses = sum(agent.game_class.reward(state, 1) == -1 for state in final_states)
        frac_losses = num_losses / float(tournament.n_games)
        results = dict()
        results['frac_losses'] = frac_losses
        results['iter'] = iter
        results['model'] = agent.evaluator
        self.evaluation_results_ready = True
        self.evaluation_results = results
        return results

    def _log(self, cur_log_dict):
        print(cur_log_dict)
        if self.wandb:
            wandb.log(cur_log_dict)

    def log_evaluation_results_if_available(self):
        if self.currently_evaluating and self.evaluation_results_ready:
            results = self.evaluation_results
            self.currently_evaluating = False
            self.evaluation_results_ready = False
            self.log_evaluation_results(results)

    def wait_for_and_log_evaluation_results(self):
        if self.currently_evaluating:
            self.evaluation_thread.join()
            results = self.evaluation_results
            self.currently_evaluating = False
            self.evaluation_results_ready = False
            self.log_evaluation_results(results)

    def log_evaluation_results(self, results):
        frac_losses = results['frac_losses']
        self._log({'iter': results['iter'],
                   'frac_loss_vs_minimax': frac_losses})

        if self.wandb:
            torch.save(results['model'].state_dict(), os.path.join(wandb.run.dir, 'most_recent_model.pt'))

            if self.best_frac_losses is None or frac_losses < self.best_frac_losses:
                self.best_frac_losses = frac_losses
                torch.save(results['model'].state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))

    def log_action_prob_plot(self, processed_state, name):
        with torch.no_grad():
            p, v = self.model(processed_state)

        p = p.cpu().numpy()
        fig, ax = plt.subplots()
        ax.bar(range(len(p)), p)
        ax.set_xticks(range(len(p)))
        ax.set_xticklabels(self.agent.game_class.ALL_ACTIONS)
        self.logger.add_figure(name, fig, self.train_iter_count.value)
        plt.close(fig)


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--single_process', action='store_true', default=False)
    parser.add_argument('-l', '--lr', type=float, default=0.005)
    parser.add_argument('--tau', type=(lambda x: [float(val) for val in x.split(',')]), default=1.)
    parser.add_argument('--c_puct', type=float, default=1., help='c_puct controls exploration level in MCTS search')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3, help='Parameter for Dirichlet noise')
    parser.add_argument('-p', '--n_self_play_procs', type=int, default=1, help='Number of self-play processes')
    parser.add_argument('-n', '--n_games_per_proc', type=int, required=True,
                        help='Number of self-play games per process')
    parser.add_argument('--n_training_workers', type=int, default=1, help='Number of self-play games per process')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--optimizer_class', type=str, default='Adam', help='optimizer class in torch.optim')
    parser.add_argument('--evaluate_every_n_iters', type=int, default=1000,
                        help='how often to play tournament against minimax')
    parser.add_argument('--n_games_in_evaluation', type=int, default=100)
    parser.add_argument('--wandboff', action='store_true', default=False)
    return parser


def run(args, model, monitor=None):
    alpha_agent_kwargs = {'training_tau': args.tau, 'c_puct': args.c_puct, 'dirichlet_alpha': args.dirichlet_alpha}
    if args.single_process:
        train_alpha_agent(TicTacToe, model, monitor=monitor, n_games=args.n_games_per_proc,
                          alpha_agent_kwargs=alpha_agent_kwargs)
    else:
        train_alpha_agent_mp(TicTacToe, model, monitor=monitor, n_self_play_workers=args.n_self_play_procs,
                             alpha_agent_kwargs=alpha_agent_kwargs,
                             n_games_per_worker=args.n_games_per_proc, n_training_workers=args.n_training_workers)


def main():
    parser = get_parser()
    args = parser.parse_args()
    mp.set_start_method('forkserver')
    optimizer_class = eval(f'torch.optim.{args.optimizer_class}')

    def optimizer(parameters):
        return optimizer_class(parameters, lr=args.lr)

    device = torch.device(args.device)
    model = TicTacToeAlphaModel(optimizer, device=device)
    monitor_log_queue = mp.Queue(maxsize=1000)
    if args.single_process:
        monitor = lambda model, agent: Monitor(model, agent, args, wandb=(not args.wandboff),
                                               evaluate_every_n_iters=args.evaluate_every_n_iters,
                                               n_games_in_evaluation=args.n_games_in_evaluation)
    else:
        monitor = lambda model, agent, kill, pause_training: Monitor(model, agent, args, monitor_log_queue,
                                                                     kill, pause_training, wandb=(not args.wandboff),
                                                                     evaluate_every_n_iters=args.evaluate_every_n_iters,
                                                                     n_games_in_evaluation=args.n_games_in_evaluation)

    run(args, model, monitor)


if __name__ == '__main__':
    main()
