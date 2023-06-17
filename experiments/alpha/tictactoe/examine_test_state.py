import argparse
from aigames.training_manager.alpha_training_manager_lightning import AlphaTrainingRunLightning
from aigames.game.tictactoe import board_string_to_array, FastTicTacToeState
from experiments.alphatoe import TicTacToeNetwork, FastTicTacToeEvaluator  # Need these for model checkpoint loading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', '-c', type=str, help='Path for model checkpoint')
    parser.add_argument('--debug', '-d', action='store_true', help='Enter debug mode after playing')
    parser.add_argument('--state', '-s', type=str, default='x--|-x-|o--', help='State to examine')
    args = parser.parse_args()

    # Load the checkpoint
    model = AlphaTrainingRunLightning.load_from_checkpoint(args.ckpt_path)
    state = FastTicTacToeState(board_string_to_array(args.state))

    # Evaluate the state
    print('State being evaluated: ')
    print(state)
    print('Network evaluation: ')
    print(model.alpha_evaluator.evaluate(state))

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
