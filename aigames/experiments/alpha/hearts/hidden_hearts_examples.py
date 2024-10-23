from aigames.agent.alpha_agent_hidden import AlphaAgentHidden, AlphaAgentHiddenHyperparameters
from aigames.agent.alpha_agent_multi import ConstantMCTSIters, DummyAlphaEvaluatorMulti
from aigames.agent.hearts.hearts_sampler import HeartsSampler
from aigames.agent.hearts.simple_hearts_agent import SimpleHeartsAgent
from aigames.agent.random_agent_multi import RandomAgentMulti
from aigames.game.command_line_game import CommandLineGame
from aigames.game.hearts import Hearts
from aigames.game.hearts_hidden import HeartsHidden
import torch

from aigames.utils.listeners import ActionCounterProgressBar, MaxActionGameKiller, RewardListenerMulti


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run Hidden Hearts game with AlphaAgent")
    parser.add_argument("--num_games", type=int, default=1000, help="Number of games to run")
    parser.add_argument("--mcts_iters", type=int, default=100, help="Number of MCTS iterations")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to the model checkpoint (optional)")

    args = parser.parse_args()

    # Validate model checkpoint path if provided
    if args.model_checkpoint and not os.path.isfile(args.model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint file not found: {args.model_checkpoint}")

    # Update hyperparameters based on arguments
    hyperparams = AlphaAgentHiddenHyperparameters(
        n_samples=10,
        n_mcts_iters=ConstantMCTSIters(args.mcts_iters)
    )

    # TODO: If a model checkpoint is provided, load it here
    # if args.model_checkpoint:
    #     # Load the model from the checkpoint
    #     # This part depends on how your model is structured and saved
    #     # Example: model = load_model(args.model_checkpoint)
    #     pass
    
    # Initialize agents
    hidden_alpha_agent = AlphaAgentHidden(
        game_class=HeartsHidden,
        evaluator=DummyAlphaEvaluatorMulti(n_actions=HeartsHidden.get_n_actions(),
                                           n_players=HeartsHidden.get_n_players(),
                                           device='cpu'),
        hyperparams=hyperparams,
        listeners=[],
        sampler=HeartsSampler()
    )
    
    hidden_alpha_agent.eval()
    
    simple_hearts_agents = [SimpleHeartsAgent() for _ in range(3)]

    # Set up the game
    all_agents = [hidden_alpha_agent] + simple_hearts_agents
    reward_listener = RewardListenerMulti(1.0)
    game = HeartsHidden(args.num_games, all_agents, [ActionCounterProgressBar(52), reward_listener])

    # Play the game
    game.play()
    
    # Print out the average reward per player
    print(reward_listener.rewards.mean(dim=0))
    
if __name__ == "__main__":
    main()