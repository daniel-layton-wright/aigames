"""

"""
from aigames.game import CommandLineGame
from aigames.game.twenty_forty_eight import TwentyFortyEight
from aigames.agent.alpha_agent import AlphaAgent, DummyAlphaEvaluator, AlphaAgentHyperparameters
from aigames.utils.utils import play_tournament
from aigames.utils.listeners import AverageRewardListener


def main():
    # Set up an arg parser to read in number of games to play
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=100)
    args = parser.parse_args()

    # Keep track of the reward for each game with a listener

    alpha_evaluator = DummyAlphaEvaluator(4)
    hyperparams = AlphaAgentHyperparameters()
    hyperparams.n_mcts = 100
    alpha_agent = AlphaAgent(TwentyFortyEight, alpha_evaluator, hyperparams)
    listener = AverageRewardListener(1, 0)
    play_tournament(TwentyFortyEight, [alpha_agent], args.n_games, tqdm=True, listeners=[listener])

    # Print out average reward results
    print(f'Average reward: {listener.avg_reward}')


if __name__ == '__main__':
    main()
