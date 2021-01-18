from aigames.game.tictactoe import TicTacToe
from aigames.agent.alpha_agent import *
from aigames.agent.minimax_agent import MinimaxAgent
from aigames.game.command_line_game import CommandLineGame
from aigames.utils.utils import play_tournament


def main():
    minimax = MinimaxAgent(TicTacToe)
    dummy = DummyAlphaEvaluator(9)
    for n_mcts in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        alpha = AlphaAgent(TicTacToe, dummy, n_mcts=n_mcts, dirichlet_epsilon=0., c_puct=1)
        alpha.eval()
        x = play_tournament(TicTacToe, [minimax, alpha], 100)
        print(n_mcts, x)


if __name__ == '__main__':
    main()