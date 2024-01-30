"""
Prints out a game between to naive MCTS players (dummy alpha evaluator)
"""
from aigames.game import CommandLineGame
from aigames.game.G2048_multi import G2048Multi
from aigames.agent.alpha_agent import AlphaAgent, DummyAlphaEvaluator, AlphaAgentHyperparameters
from aigames.agent.alpha_agent_multi import AlphaAgentMulti, AlphaAgentHyperparametersMulti as AlphaAgentHyperparametersMulti, DummyAlphaEvaluatorMulti
from .network_architectures import TwentyFortyEightNetwork, TwentyFortyEightEvaluator
from aigames.utils.utils import play_tournament_multi
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
        alpha_evaluator = DummyAlphaEvaluatorMulti(4, 1)

    hyperparams = AlphaAgentHyperparametersMulti()
    hyperparams.n_mcts = 100
    alpha_agent = AlphaAgentMulti(G2048Multi, alpha_evaluator, hyperparams)

    listeners = []
    if not args.hide_game:
        listeners = [CommandLineGame(pause_time=0.1)]

    play_tournament_multi(G2048Multi, alpha_agent, args.n_games, 1, listeners=listeners)


if __name__ == '__main__':
    main()
