"""
Get some baseline scores for the hearts game against a simple heuristic agent
"""
from aigames.game.hearts import Hearts
from aigames.agent.random_agent_multi import RandomAgentMulti
from aigames.agent.hearts.simple_hearts_agent import SimpleHeartsAgent
from aigames.utils.listeners import RewardListenerMulti, ActionCounterProgressBar
from aigames.utils.utils import play_tournament_multi


def main():
    """
    Run baseline experiments for Hearts game and write results to a file.
    """
    import argparse
    import pandas as pd
    import datetime
    import subprocess
    from aigames.agent.alpha_agent_multi import AlphaAgentMulti, DummyAlphaEvaluatorMulti, AlphaAgentHyperparametersMulti

    # Set up argparser to specify argument, such as the number of games to play
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=1000)
    args = parser.parse_args()

    # Get current timestamp and git status
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    git_status = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    results = []

    # Random Agent vs Simple Heuristic Agents
    listener = RewardListenerMulti(1, False)
    play_tournament_multi(Hearts,
                          [RandomAgentMulti(Hearts), SimpleHeartsAgent(), SimpleHeartsAgent(), SimpleHeartsAgent()],
                          args.n_games, 1, tqdm=True, listeners=[listener])

    df = pd.DataFrame(listener.rewards)
    results.append(("Random Agent vs Simple Heuristic Agents", df.describe()))

    # AlphaAgent with dummy evaluator vs Simple Heuristic Agents
    eval = DummyAlphaEvaluatorMulti(52, 4)
    hypers = AlphaAgentHyperparametersMulti()
    alpha_agent = AlphaAgentMulti(Hearts, eval, hypers)
    listener = RewardListenerMulti(1, False)
    play_tournament_multi(Hearts, [alpha_agent, SimpleHeartsAgent(), SimpleHeartsAgent(), SimpleHeartsAgent()],
                          args.n_games, 1, tqdm=True,
                          listeners=[listener, ActionCounterProgressBar(52,
                                                                        'AlphaAgent vs Simple Heuristic Agents')])

    df = pd.DataFrame(listener.rewards)
    results.append(("AlphaAgent with dummy evaluator vs Simple Heuristic Agents", df.describe()))
    
    # AlphaAgent with Gumbel action selector vs Simple Heuristic Agents
    from aigames.mcts.mcts import GumbelActionSelector, GumbelHyperparameters

    gumbel_eval = DummyAlphaEvaluatorMulti(52, 4)
    gumbel_hypers = AlphaAgentHyperparametersMulti(action_selector=GumbelActionSelector(GumbelHyperparameters()), dirichlet_epsilon=0.0)
    gumbel_alpha_agent = AlphaAgentMulti(Hearts, gumbel_eval, gumbel_hypers)
    gumbel_listener = RewardListenerMulti(1, False)
    play_tournament_multi(Hearts, [gumbel_alpha_agent, SimpleHeartsAgent(), SimpleHeartsAgent(), SimpleHeartsAgent()],
                          args.n_games, 1, tqdm=True,
                          listeners=[gumbel_listener, ActionCounterProgressBar(52,
                                                                               'AlphaAgent with Gumbel vs Simple Heuristic Agents')])

    gumbel_df = pd.DataFrame(gumbel_listener.rewards)
    results.append(("AlphaAgent with Gumbel action selector vs Simple Heuristic Agents", gumbel_df.describe()))

    # Write results to file
    with open(f'hearts_baseline_results_{timestamp}.txt', 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Git Hash: {git_hash}\n")
        f.write(f"Git Status:\n{git_status}\n\n")
        f.write(f"Number of games: {args.n_games}\n\n")
        for title, result in results:
            f.write(f"{title}\n")
            f.write(f"{result}\n\n")

    print(f"Results written to hearts_baseline_results_{timestamp}.txt")


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
