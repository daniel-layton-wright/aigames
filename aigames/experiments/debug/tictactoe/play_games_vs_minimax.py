import sys
sys.path.insert(0, '/')
sys.path.insert(0, '/aigames/experiments')

from alphatoe import *


def main():
    network = TicTacToeNetwork()
    evaluator = FastTicTacToeEvaluator(network)
    agent = AlphaAgent(FastTicTacToe, evaluator, AlphaAgentHyperparameters())
    minimax = MinimaxAgent(FastTicTacToe)
    # cli = CommandLineGame()
    game = FastTicTacToe([minimax, agent],
                         # listeners=[cli]
                         )

    for _ in tqdm(range(100)):
        game.play()


if __name__ == '__main__':
    main()