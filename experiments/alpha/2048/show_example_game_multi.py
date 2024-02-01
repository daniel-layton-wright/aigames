"""
Prints out a game between to naive MCTS players (dummy alpha evaluator)
"""
from aigames.game import CommandLineGame
from aigames.game.G2048_multi import get_G2048Multi_game_class
from aigames.agent.alpha_agent import AlphaAgent, DummyAlphaEvaluator, AlphaAgentHyperparameters
from aigames.agent.alpha_agent_multi import AlphaAgentMulti, AlphaAgentHyperparametersMulti as AlphaAgentHyperparametersMulti, DummyAlphaEvaluatorMulti
from aigames.utils.listeners import AvgRewardListenerMulti, ActionCounterProgressBar
from .network_architectures import TwentyFortyEightNetwork, TwentyFortyEightEvaluator
from aigames.utils.utils import play_tournament_multi
import argparse


def main():
    # Set up an argparser
    parser = argparse.ArgumentParser()
    # Add argument for device
    parser.add_argument('--device', type=str, default='cpu')
    # Add an argument for whether to use a network or not
    parser.add_argument('--use_network', action='store_true')
    # Add an argument for the number of games
    parser.add_argument('--n_games', type=int, default=1)
    # Add an argument for hiding the command line game interface if desired
    parser.add_argument('--hide_game', action='store_true')
    # Add argument for whether to show average score listener
    parser.add_argument('--show_avg_score', action='store_true')
    # Add argument for action counter
    parser.add_argument('--show_action_counter', action='store_true')
    # Pdb flag
    parser.add_argument('--pdb', action='store_true')
    # Parse the args
    args = parser.parse_args()

    G2048Multi = get_G2048Multi_game_class(args.device)

    # If using a network set up the network, otherwise use a dummy evaluator
    if args.use_network:
        network = TwentyFortyEightNetwork()
        network.eval()
        alpha_evaluator = TwentyFortyEightEvaluator(network)
    else:
        alpha_evaluator = DummyAlphaEvaluatorMulti(4, 1, args.device)

    hyperparams = AlphaAgentHyperparametersMulti()
    hyperparams.mcts_hyperparams.n_iters = 100
    alpha_agent = AlphaAgentMulti(G2048Multi, alpha_evaluator, hyperparams)

    listeners = []
    if not args.hide_game:
        listeners = [CommandLineGame(pause_time=0.1)]

    if args.show_avg_score:
        listeners.append(AvgRewardListenerMulti(hyperparams.discount_rate, 0, show_tqdm=True, tqdm_total=20000))

    if args.show_action_counter:
        listeners.append(ActionCounterProgressBar(200))

    play_tournament_multi(G2048Multi, alpha_agent, args.n_games, 1, listeners=listeners)

    if args.pdb:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
