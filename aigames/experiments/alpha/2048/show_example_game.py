"""
Prints out a game between to naive MCTS players (dummy alpha evaluator)
"""
from aigames.game import CommandLineGame
from aigames.game.twenty_forty_eight import TwentyFortyEight
from aigames.agent.alpha_agent import AlphaAgent, DummyAlphaEvaluator, AlphaAgentHyperparameters
from .network_architectures import TwentyFortyEightNetwork, TwentyFortyEightEvaluator
from aigames.utils.utils import play_tournament
import argparse


def main():
    # Set up an argparser
    parser = argparse.ArgumentParser()
    # Add an argument for whether to use a network or not
    parser.add_argument('--use_network', action='store_true')
    # Add an argument for the number of games
    parser.add_argument('--n_games', type=int, default=1)
    # Add an argument for hiding the command line game interface if desired
    parser.add_argument('--hide_game', action='store_true')
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
    hyperparams.n_mcts = 100
    alpha_agent = AlphaAgent(TwentyFortyEight, alpha_evaluator, hyperparams)

    listeners = []
    if not args.hide_game:
        listeners = [CommandLineGame(pause_time=0.1)]

    play_tournament(TwentyFortyEight, [alpha_agent], args.n_games, listeners=listeners)


if __name__ == '__main__':
    main()
