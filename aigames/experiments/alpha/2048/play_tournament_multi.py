"""

"""
from aigames.game.G2048_multi import G2048Multi
from aigames.agent.alpha_agent_multi import AlphaAgentMulti, DummyAlphaEvaluatorMulti, AlphaAgentHyperparametersMulti
from aigames.utils.utils import play_tournament_multi
from aigames.utils.listeners import AverageRewardListener


def main():
    # Set up an arg parser to read in number of games to play
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=100)
    args = parser.parse_args()

    # Keep track of the reward for each game with a listener
    alpha_evaluator = DummyAlphaEvaluatorMulti(4, 1)
    hyperparams = AlphaAgentHyperparametersMulti()
    hyperparams.n_mcts = 100
    hyperparams.mcts_hyperparams.c_puct = 50
    alpha_agent = AlphaAgentMulti(G2048Multi, alpha_evaluator, hyperparams)
    listener = AverageRewardListener(1, 0)
    play_tournament_multi(G2048Multi, alpha_agent, args.n_games, 1, tqdm=False, listeners=[])


if __name__ == '__main__':
    main()
