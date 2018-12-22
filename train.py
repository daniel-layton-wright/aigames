from manual_agent import *
from minimax_agent import *
from qlearning_agent import *
from tictactoe import *
import torch.nn as nn


def main():
    minimax_agent = MinimaxAgent(TicTacToe)
    manual_agent = ManualTTTAgent()
    q = Q(TicTacToe, 100, nn.ReLU)
    learning_agent = QLearningAgent(TicTacToe, q, lr = 0.005, exploration_probability=0.3)

    for i in range(100000):
        game = TicTacToe([minimax_agent, minimax_agent], verbose = False)
        game.play()

        if i % 1000 == 0:
            # Play the minimax agent and record the average
            n_games = 100
            n_losses = 0
            for _ in range(n_games):
                learning_agent.training = False
                game = TicTacToe([minimax_agent, learning_agent], verbose=False)
                game.play()

                if game.reward(game.state, 1) == game.LOSE_REWARD:
                    n_losses += 1

            print('Percent Losses against Minimax: {}%'.format(100.0 * n_losses / n_games))

            learning_agent.training = True

if __name__ == "__main__":
    main()


"""
Notes: (with 100 hidden size)
lr = 0.001, exploration_probability works reasonably well when training against minimax_agent, still lingering losses down to ~9% losses after 7000 games
lr = 0.005 seems to work okay when self-playing


"""