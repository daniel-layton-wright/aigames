import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from aigames import *


class Monitor:
    def __init__(self):
        self.score_for_winning_position_history = []
        self.n = 0

    def debug(self, game):
        winning_state = np.array([[-1, -1, 0], [1, 1, 0], [0, 0, 1]])
        Q = game.players[1].Q
        scores = []
        legal_actions = game.legal_actions(winning_state)
        for action in legal_actions:
            processed_state_action = Q.process_state_action(winning_state, action, game.get_player_index(winning_state))
            with torch.no_grad():
                scores.append(Q(processed_state_action))

        self.score_for_winning_position_history.append(max(scores))

        if len(self.score_for_winning_position_history) % 100 == 0:
            plt.plot(self.score_for_winning_position_history)
            plt.savefig('score_for_winning_position_history.png')
            plt.close()
            #
            # print(game.state_to_str(winning_state))
            # legal_actions = game.legal_actions(winning_state)
            # legal_action_indices = [game.ALL_ACTIONS.index(a) for a in legal_actions]
            # legal_scores = pred_scores[legal_action_indices]
            #
            # for action, score in zip(legal_actions, legal_scores):
            #     print(action, ': ', score)

        self.n += 1



def train(lr = 0.005, exploration_probability = 0.1, n_hidden = 100, activation_fn = nn.ReLU, batch_size = 32,
          checkpoint_path = None, n_iters = 100000, update_target_Q_every = 1000):
    minimax_agent = MinimaxAgent(TicTacToe)
    manual_agent = ManualAgent()
    q0 = Q(TicTacToe, n_hidden, activation_fn)
    q = Q(TicTacToe, n_hidden, activation_fn)
    monitor = Monitor()
    learning_agent0 = QLearningAgent(TicTacToe, q0, lr = lr, exploration_probability=exploration_probability,
                                     batch_size = batch_size, update_target_Q_every = update_target_Q_every)
    learning_agent = QLearningAgent(TicTacToe, q, lr = lr, exploration_probability=exploration_probability,
                                    batch_size = batch_size, update_target_Q_every = update_target_Q_every)
    pct_loss_vs_minimax_history = []
    start_iter = 0

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        learning_agent.Q.load_state_dict(checkpoint['model_state_dict'])
        learning_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        learning_agent.loss_history = checkpoint['loss_history']
        learning_agent.loss_ema_history = checkpoint['loss_ema_history']
        learning_agent.replay_memory = checkpoint['replay_memory']
        start_iter = checkpoint['iter']

    update_every = 1000

    for i in range(start_iter, n_iters):
        game = TicTacToe([learning_agent0, learning_agent], verbose = False, monitor=monitor.debug)
        game.play()

        if (i + 1) % update_every == 0:
            # Play the minimax agent and record the average
            n_games = 100
            n_losses = 0

            for j in range(n_games):
                learning_agent.training = False
                game = TicTacToe([minimax_agent, learning_agent], verbose = False, pause_seconds=0)
                game.play()

                if game.reward(game.state, 1) == game.LOSE_REWARD:
                    n_losses += 1

            pct_loss_vs_minimax_history.append(100.0 * n_losses / n_games)
            print('Percent Losses against Minimax: {}%'.format(100.0 * n_losses / n_games))

            learning_agent.training = True

            plt.plot(learning_agent.loss_history)
            plt.plot(learning_agent.loss_ema_history)
            plt.savefig('loss_history.png')
            plt.close()
            plt.plot(pct_loss_vs_minimax_history)
            plt.savefig('vs_minimax.png')
            plt.close()

            if checkpoint_path is not None:
                torch.save({
                    'iter': i,
                    'model_state_dict': q.state_dict(),
                    'optimizer_state_dict': learning_agent.optimizer.state_dict(),
                    'loss_history': learning_agent.loss_history,
                    'loss_ema_history': learning_agent.loss_ema_history,
                    'pct_loss_vs_minimax_history': pct_loss_vs_minimax_history,
                    'replay_memory': learning_agent.replay_memory
                }, checkpoint_path)

            print(i)

    return minimax_agent, learning_agent


def main():
    import argparse
    parser = argparse.ArgumentParser(description = 'Run training')
    parser.add_argument('-l', '--lr', type = float, default = 0.005)
    parser.add_argument('-e', '--exploration_probability', type = float, default = 0.1)
    parser.add_argument('-n', '--n_hidden', type = str, default = '100')
    parser.add_argument('-a', '--activation_fn', type = str, default = 'nn.ReLU')
    parser.add_argument('-b', '--batch_size', type = int, default = 32)
    parser.add_argument('-c', '--checkpoint_path', type = str, default = None)
    args = parser.parse_args()
    activation_fn = eval(args.activation_fn)
    n_hidden = eval(args.n_hidden)

    train(lr = args.lr, exploration_probability=args.exploration_probability, n_hidden=n_hidden,
          activation_fn=activation_fn, batch_size=args.batch_size, checkpoint_path=args.checkpoint_path)


if __name__ == "__main__":
    main()


"""
Notes: (with 100 hidden size)
lr = 0.001, exploration_probability works reasonably well when training against minimax_agent, still lingering losses down to ~9% losses after 7000 games
lr = 0.005 seems to work okay when self-playing


"""