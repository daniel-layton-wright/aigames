"""
Prints out a game between to naive MCTS players (dummy alpha evaluator)
"""
from aigames.game import CommandLineGame
from aigames.game.twenty_forty_eight import TwentyFortyEight
from aigames.agent.alpha_agent import AlphaAgent, DummyAlphaEvaluator, AlphaAgentHyperparameters
from .network_architectures import TwentyFortyEightNetwork, TwentyFortyEightEvaluator
import argparse


def main():
    # Set up an argparser
    parser = argparse.ArgumentParser()
    # Add an argument for whether to use a network or not
    parser.add_argument('--use_network', action='store_true')
    # Parse the args
    args = parser.parse_args()

    # If using a network set up the network, othereise use a dummy evaluator
    if args.use_network:
        network = TwentyFortyEightNetwork()
        network.eval()
        alpha_evaluator = TwentyFortyEightEvaluator(network)
    else:
        alpha_evaluator = DummyAlphaEvaluator(4)

    hyperparams = AlphaAgentHyperparameters()
    hyperparams.n_mcts = 500
    alpha_agent = AlphaAgent(TwentyFortyEight, alpha_evaluator, hyperparams)

    game = TwentyFortyEight([alpha_agent], listeners=[CommandLineGame(pause_time=0.1)])
    game.play()


if __name__ == '__main__':
    main()
