"""
Run a tournament between the naive (untrained) alpha network and itself (useful for profiling purposes)
"""
import argparse
from experiments.alpha.connect4.network_architectures import Connect4Network
from .connect4_lightning import Connect4Evaluator
from aigames.agent.alpha_agent import AlphaAgent, AlphaAgentHyperparameters
from aigames.utils.utils import play_tournament, load_from_arg_parser, add_all_slots_to_arg_parser
from aigames.game.connect4 import Connect4
from aigames.game.command_line_game import CommandLineGame


def main():
    hyperparams = AlphaAgentHyperparameters()
    hyperparams.n_mcts = 100

    # Setup arg parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_games', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    add_all_slots_to_arg_parser(parser, hyperparams)
    args = parser.parse_args()

    load_from_arg_parser(args, hyperparams)

    # Setup the network, evaluator, and agent
    network = Connect4Network()
    evaluator = Connect4Evaluator(network)
    agent = AlphaAgent(Connect4, evaluator, hyperparams)
    # command_line_listener = CommandLineGame()

    play_tournament(Connect4, [agent, agent], args.n_games, 1, tqdm=True)

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
