from aigames.agent.alpha_agent_hidden import AlphaAgentHidden, AlphaAgentHiddenHyperparameters
from aigames.agent.alpha_agent_multi import ConstantMCTSIters, DummyAlphaEvaluatorMulti
from aigames.agent.hearts.hearts_sampler import HeartsSampler
from aigames.agent.hearts.simple_hearts_agent import SimpleHeartsAgent
from aigames.agent.random_agent_multi import RandomAgentMulti
from aigames.game.command_line_game import CommandLineGame
from aigames.game.hearts import Hearts
from aigames.game.hearts_hidden import HeartsHidden
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


from aigames.utils.listeners import ActionCounterProgressBar, MaxActionGameKiller, RewardListenerMulti


def run_experiment(num_games, mcts_iters, n_samples, model_checkpoint=None):
    """
    Run a single Hearts game experiment with specified hyperparameters.

    :param num_games: Number of games to run.
    :param mcts_iters: Number of MCTS iterations.
    :param n_samples: Number of samples to generate for each state.
    :param model_checkpoint: Path to the model checkpoint (optional).
    :return: Average reward for the first player.
    """
    # Validate model checkpoint path if provided
    if model_checkpoint and not os.path.isfile(model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint file not found: {model_checkpoint}")

    # Configure hyperparameters
    hyperparams = AlphaAgentHiddenHyperparameters(
        n_samples=n_samples,
        n_mcts_iters=ConstantMCTSIters(mcts_iters)
    )

    # TODO: Load model checkpoint if provided
    if model_checkpoint:
        # Example:
        # hidden_alpha_agent.load_model(model_checkpoint)
        pass

    # Initialize agents
    hidden_alpha_agent = AlphaAgentHidden(
        game_class=HeartsHidden,
        evaluator=DummyAlphaEvaluatorMulti(
            n_actions=HeartsHidden.get_n_actions(),
            n_players=HeartsHidden.get_n_players(),
            device='cpu'
        ),
        hyperparams=hyperparams,
        listeners=[],
        sampler=HeartsSampler()
    )
    
    hidden_alpha_agent.eval()
    
    simple_hearts_agents = [SimpleHeartsAgent() for _ in range(3)]

    # Set up the game
    all_agents = [hidden_alpha_agent] + simple_hearts_agents
    reward_listener = RewardListenerMulti(1.0)
    game = HeartsHidden(
        num_games, 
        all_agents, 
        [ActionCounterProgressBar(52), reward_listener]
    )

    # Play the game
    game.play()
    
    # Calculate and return the average reward for the first player
    average_reward = reward_listener.rewards.mean(dim=0)[0].item()
    return average_reward


def main():
    """
    Execute a grid of Hearts game experiments varying the number of samples and MCTS iterations.
    Results are saved to a CSV file and a heatmap is generated to visualize the average rewards.
    """
    parser = argparse.ArgumentParser(description="Run grid of Hidden Hearts games with AlphaAgent")
    parser.add_argument("--num_games", type=int, default=1000, help="Number of games to run per experiment")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to the model checkpoint (optional)")
    parser.add_argument("--output_csv", type=str, default="results.csv", help="Path to save the results CSV")
    parser.add_argument("--heatmap_path", type=str, default="heatmap.png", help="Path to save the heatmap image")
    
    args = parser.parse_args()

    # Define grid of hyperparameters
    n_samples_list = [10, 50, 100, 200]
    mcts_iters_list = [0, 50, 100, 200]

    # Initialize results list
    results = []

    # Iterate over the grid of hyperparameters
    for n_samples in n_samples_list:
        for mcts_iters in mcts_iters_list:
            print(f"Running experiment with n_samples={n_samples}, mcts_iters={mcts_iters}")
            try:
                avg_reward = run_experiment(
                    num_games=args.num_games,
                    mcts_iters=mcts_iters,
                    n_samples=n_samples,
                    model_checkpoint=args.model_checkpoint
                )
                results.append({
                    "n_samples": n_samples,
                    "mcts_iters": mcts_iters,
                    "average_reward_player_1": avg_reward
                })
                print(f"Average Reward for Player 1: {avg_reward}")
            except Exception as e:
                print(f"Experiment failed for n_samples={n_samples}, mcts_iters={mcts_iters}: {e}")
                results.append({
                    "n_samples": n_samples,
                    "mcts_iters": mcts_iters,
                    "average_reward_player_1": None,
                    "error": str(e)
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Pivot the DataFrame for heatmap
    heatmap_data = results_df.pivot("mcts_iters", "n_samples", "average_reward_player_1")

    # Save results to CSV
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Average Reward for Player 1")
    plt.xlabel("Number of Samples")
    plt.ylabel("Number of MCTS Iterations")
    plt.tight_layout()
    plt.savefig(args.heatmap_path)
    plt.close()
    print(f"Heatmap saved to {args.heatmap_path}")


if __name__ == "__main__":
    main()