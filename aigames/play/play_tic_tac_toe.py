from aigames.agent import ManualAgent, MinimaxAgent
from aigames.game.tictactoe import TicTacToe
from aigames.game import CommandLineGame


def main():
    cli = CommandLineGame()
    agents = [ManualAgent(), MinimaxAgent(TicTacToe)]
    tictactoe = TicTacToe(agents, [cli])
    tictactoe.play()


if __name__ == '__main__':
    main()