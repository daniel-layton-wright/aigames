"""
Run a game between a model checkpoint and the manual agent
"""
import argparse
from experiments.alpha.connect4.connect4_lightning import AlphaTrainingRunLightningConnect4, Connect4Evaluator
from aigames.agent.manual_agent import ManualAgent
from aigames.game.connect4 import Connect4
from aigames.game.command_line_game import CommandLineGame


def main():
    # Set up argparser to load model checkpoint
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--model_self_play', '-m', action='store_true', help='Whether to use the model for self play')
    args = parser.parse_args()

    # Load the
    checkpoint_run = AlphaTrainingRunLightningConnect4.load_from_checkpoint(args.ckpt)

    alpha_agent = checkpoint_run.agents[-1]
    alpha_agent.eval()  # Put the agent in eval mode

    # Set up the game
    if not args.model_self_play:
        manual_agent = ManualAgent()
        agents = [manual_agent, alpha_agent]
    else:
        agents = [alpha_agent, alpha_agent]

    game = Connect4(agents, [CommandLineGame(pause_time=2)])
    game.play()


if __name__ == '__main__':
    main()
