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


"""
Random Agent vs Simple Heuristic Agents
                 0           1            2            3
count  1000.000000  1000.00000  1000.000000  1000.000000
mean    -11.175000    -4.87400    -5.298000    -5.589000
std       7.891672     6.32473     6.831538     6.890834
min     -26.000000   -26.00000   -26.000000   -26.000000
25%     -18.000000    -7.00000    -8.000000    -8.000000
50%     -10.000000    -2.00000    -2.000000    -3.000000
75%      -4.000000     0.00000     0.000000     0.000000
max       0.000000     0.00000     0.000000     0.000000

AlphaAgent with dummy evaluator vs Simple Heuristic Agents
                 0            1            2           3
count  1000.000000  1000.000000  1000.000000  1000.00000
mean     -9.341000    -6.273000    -5.926000    -6.12400
std       8.042102     7.475593     7.221543     7.30041
min     -26.000000   -26.000000   -26.000000   -26.00000
25%     -17.000000   -12.000000    -9.000000   -10.00000
50%      -7.000000    -3.000000    -3.000000    -3.00000
75%      -2.000000     0.000000     0.000000     0.00000
max       0.000000     0.000000     0.000000     0.00000
"""
