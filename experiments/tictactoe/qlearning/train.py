import os
import numpy as np
import torch.nn as nn
import torch
import sys
top_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))
sys.path.insert(0, top_dir)
from aigames import *
import hypertune


class TicTacToeQNetwork(nn.Module):
    def __init__(self, n_filters = (64, 16)):
        super().__init__()
        self.game = TicTacToe
        self.network = nn.Sequential(
            torch.nn.Conv2d(3, n_filters[0], 3, stride=1, padding=1),
            nn.ReLU(),
            torch.nn.Conv2d(n_filters[0], n_filters[1], 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=(n_filters[1] * 3 * 3), out_features=1, bias=True)
        )

    def forward(self, processed_state):
        return self.network(processed_state)

    def process_state_action(self, state, action, player_index):
        x = self.game.get_next_state(state, action)
        x = torch.tensor(x).to(torch.float)
        x = x.unsqueeze(0)
        return x


class TicTacToeMonitor(QLearningMonitor):
    def __init__(self, evaluate_every_n_games, job_dir):
        self.score_for_winning_position_history = []
        self.evaluate_every_n_games = evaluate_every_n_games
        self.job_dir = job_dir
        self.pct_loss_vs_minimax_history = []
        self.minimax_agent = MinimaxAgent(TicTacToe)

    def on_game_end(self, qlearning_agent, game_number):

        if game_number % self.evaluate_every_n_games == 0:
            # Play the minimax agent and record the average
            n_games = 200
            n_losses = 0

            qlearning_agent.eval()
            for j in range(n_games):
                game = TicTacToe([self.minimax_agent, qlearning_agent], verbose = False, pause_seconds=0)
                game.play()

                if game.reward(game.state, 1) == game.LOSE_REWARD:
                    n_losses += 1

            qlearning_agent.train()

            pct_loss_vs_minimax = 100.0 * n_losses / n_games
            self.pct_loss_vs_minimax_history.append(pct_loss_vs_minimax)
            print('Percent Losses against Minimax: {}%'.format(100.0 * n_losses / n_games))

            if pct_loss_vs_minimax == max(self.pct_loss_vs_minimax_history):
                # this is the best model
                save_checkpoint(self.job_dir, game_number, qlearning_agent, 'best.pt')

            save_checkpoint(self.job_dir, game_number, qlearning_agent, 'latest.pt')

            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='pct_loss_vs_minimax',
                metric_value=pct_loss_vs_minimax,
                global_step=game_number)


        winning_state = np.zeros((3,3,3)).astype(int)
        winning_state[0,1,0] = 1
        winning_state[0,1,1] = 1
        winning_state[1,0,0] = 1
        winning_state[1,0,1] = 1
        winning_state[2,:,:] = 1
        Q = qlearning_agent.Q
        scores = []
        legal_actions = TicTacToe.legal_actions(winning_state)
        for action in legal_actions:
            processed_state_action = Q.process_state_action(winning_state, action, TicTacToe.get_player_index(winning_state))
            with torch.no_grad():
                scores.append(Q(processed_state_action))

        self.score_for_winning_position_history.append(max(scores))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('-l', '--lr', type=float, default=0.005)
    parser.add_argument('-e', '--exploration_probability', type=float, default=0.1)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--job-dir', type=str, default=None)
    parser.add_argument('-u', '--update_target_Q_every', type=int, default=10000)
    parser.add_argument('--min_replay_memory_size', type=int, default=10000)
    parser.add_argument('--max_replay_memory_size', type=int, default=50000)
    parser.add_argument('-n', '--n_games', type=int, default=100000)
    args = parser.parse_args()

    q = TicTacToeQNetwork()
    qlearning_agent = QLearningAgent(TicTacToe, q,
                                     lr=args.lr, exploration_probability=args.exploration_probability,
                                     batch_size=args.batch_size, update_target_Q_every=args.update_target_Q_every,
                                     min_replay_memory_size=args.min_replay_memory_size,
                                     max_replay_memory_size=args.max_replay_memory_size)

    start_iter = 0
    if args.job_dir is not None:
        checkpoint_path = os.path.join(args.job_dir, 'latest.pt')
        start_iter = load_checkpoint(checkpoint_path, qlearning_agent)

    monitor = TicTacToeMonitor(evaluate_every_n_games=1000, job_dir=args.job_dir)

    train_qlearning_agent(TicTacToe, qlearning_agent=qlearning_agent, monitor=monitor,
                          start_iter=start_iter, n_games=args.n_games)


if __name__ == "__main__":
    main()


"""
Notes: (with 100 hidden size)
lr = 0.001, exploration_probability works reasonably well when training against minimax_agent, still lingering losses down to ~9% losses after 7000 games
lr = 0.005 seems to work okay when self-playing


"""