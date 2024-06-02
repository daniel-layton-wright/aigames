"""
Get some baseline scores for the hearts game against a simple heuristic agent
"""
from aigames.game.hearts import Hearts
from aigames.agent.random_agent_multi import RandomAgentMulti
from aigames.agent.hearts.simple_hearts_agent import SimpleHeartsAgent
from aigames.utils.listeners import RewardListenerMulti, ActionCounterProgressBar
from aigames.utils.utils import play_tournament_multi


def main():
    # Set up argparser to specify argument, such as the number of games to play
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=1000)
    args = parser.parse_args()

    listener = RewardListenerMulti(1, False)
    play_tournament_multi(Hearts,
                          [RandomAgentMulti(Hearts), SimpleHeartsAgent(), SimpleHeartsAgent(), SimpleHeartsAgent()],
                          args.n_games, 1, tqdm=False, listeners=[listener])

    # Put the listener.rewards in a dataframe and print out summary
    import pandas as pd
    df = pd.DataFrame(listener.rewards)
    print('Random Agent vs Simple Heuristic Agents')
    print(df.describe())

    # Now play an AlphaAgent with dummy evaluator against simple heuristic agents
    from aigames.agent.alpha_agent_multi import AlphaAgentMulti, DummyAlphaEvaluatorMulti, AlphaAgentHyperparametersMulti
    eval = DummyAlphaEvaluatorMulti(52, 4)
    hypers = AlphaAgentHyperparametersMulti()
    alpha_agent = AlphaAgentMulti(Hearts, eval, hypers)
    listener = RewardListenerMulti(1, False)
    play_tournament_multi(Hearts, [alpha_agent, SimpleHeartsAgent(), SimpleHeartsAgent(), SimpleHeartsAgent()],
                          args.n_games, 1, tqdm=False,
                          listeners=[listener, ActionCounterProgressBar(52,
                                                                        'AlphaAgent vs Simple Heuristic Agents')])

    # Put the listener.rewards in a dataframe and print out summary
    df = pd.DataFrame(listener.rewards)
    print('AlphaAgent with dummy evaluator vs Simple Heuristic Agents')
    print(df.describe())


if __name__ == '__main__':
    main()
