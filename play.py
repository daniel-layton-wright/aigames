from manual_agent import *
from minimax_agent import *
from qlearning_agent import *
from tictactoe import *


def main():
    minimax_agent = MinimaxAgent(TicTacToe)
    manual_agent = ManualTTTAgent()
    learning_agent = QLearningAgent(TicTacToe, Q(TicTacToe, 100))
    game = TicTacToe([manual_agent, learning_agent], verbose = True)
    game.play()

if __name__ == "__main__":
    main()
