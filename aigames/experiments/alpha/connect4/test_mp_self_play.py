import argparse
from aigames.game.connect4 import Connect4V2
from aigames.training_manager.alpha_training_manager_mp import AlphaTrainingMPData, AlphaEvaluatorMP, AlphaSelfPlayMP,\
    AlphaTrainingHyperparametersLightningMP, AlphaTrainingRunLightningMP
from aigames.utils.utils import add_all_slots_to_arg_parser, load_from_arg_parser
from .network_architectures import Connect4NetworkV2, Connect4EvaluatorV2


def main():
    # Setup the hyperparameters and argparser
    hyperparameters = AlphaTrainingHyperparametersLightningMP()
    parser = argparse.ArgumentParser()
    add_all_slots_to_arg_parser(parser, hyperparameters)
    parser.add_argument('--n_games', '-n', type=int, help='Number of games to play', default=20)
    parser.add_argument('--debug', '-d', action='store_true', help='Open PDB at the end')
    args = parser.parse_args()

    # Load the hyperparameters from the argparser
    load_from_arg_parser(args, hyperparameters)

    # Setup the network and evaluator
    network = Connect4NetworkV2()
    evaluator = Connect4EvaluatorV2(network)

    # Setup the training class
    alpha_training_mp = AlphaTrainingRunLightningMP(Connect4V2, evaluator, hyperparameters)

    alpha_training_mp.mp_self_play(args.n_games)

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
