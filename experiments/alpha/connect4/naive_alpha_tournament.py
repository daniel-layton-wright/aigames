"""
Run a tournament between the naive (untrained) alpha network and itself (useful for profiling purposes)
"""
import argparse
from experiments.alpha.connect4.network_architectures import Connect4Network
from .network_architectures import Connect4Evaluator, Connect4EvaluatorV2, Connect4NetworkV2
from aigames.agent.alpha_agent import AlphaAgent, AlphaAgentHyperparameters
from aigames.utils.utils import play_tournament, load_from_arg_parser, add_all_slots_to_arg_parser
from aigames.game.connect4 import Connect4, Connect4V2
from aigames.game.command_line_game import CommandLineGame
from aigames.training_manager.alpha_training_manager_lightning import AlphaTrainingRunLightning, AlphaTrainingHyperparametersLightning
import sys


def main():
    hyperparams = AlphaTrainingHyperparametersLightning()
    hyperparams.n_mcts = 100

    # Setup arg parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_games', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--game_class', type=str, default='Connect4V2')
    parser.add_argument('--evaluator_class', type=str, default='Connect4EvaluatorV2')
    parser.add_argument('--network_class', type=str, default='Connect4NetworkV3')
    add_all_slots_to_arg_parser(parser, hyperparams)
    args = parser.parse_args()

    load_from_arg_parser(args, hyperparams)

    # Get the game class from the class name
    game_class = getattr(sys.modules['aigames.game.connect4'], args.game_class)
    evaluator_class = getattr(sys.modules['experiments.alpha.connect4.network_architectures'], args.evaluator_class)
    network_class = getattr(sys.modules['experiments.alpha.connect4.network_architectures'], args.network_class)

    # Setup the network, evaluator, and agent
    network = network_class()
    evaluator = evaluator_class()
    alpha_run = AlphaTrainingRunLightning(game_class, network, evaluator, hyperparams)
    agent = AlphaAgent(game_class, alpha_run.alpha_evaluator, hyperparams)
    # command_line_listener = CommandLineGame()

    network.eval()
    play_tournament(game_class, [agent, agent], args.n_games, 1, tqdm=True)

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
