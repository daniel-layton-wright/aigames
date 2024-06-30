import torch

from aigames.mcts.mcts import MCTS
from aigames.mcts.utils import graph_from_mcts
from aigames.training_manager.alpha_training_manager_multi_lightning import AlphaMultiTrainingRunLightning
from aigames.utils.utils import import_string


def main():
    # Set up argparser to load checkpoint and state to use
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, help='The checkpoint to load')
    parser.add_argument('--state_path', type=str, help='The state to use for MCTS diagnostics')
    parser.add_argument('--training_run_class', type=str, help='The training run class to use')
    parser.add_argument('--graph_path', type=str, help='The path to save the graph to')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, help='Open pdb at the end')
    args = parser.parse_args()

    # Load the checkpoint
    training_run_class = import_string(args.training_run_class)
    training_run: AlphaMultiTrainingRunLightning = training_run_class.load_from_checkpoint(args.ckpt_path,
                                                                                           map_location='cpu')

    # Load the states
    states = torch.load(args.state_path)

    # Run mcts and make the graph
    game = training_run.game
    training_run.agent.eval()

    mcts = MCTS(game, training_run.agent.evaluator, training_run.hyperparams, 100, states)
    mcts.search_for_n_iters(100)
    graph = graph_from_mcts(mcts)
    graph.draw(args.graph_path, prog='dot')

    training_run.agent.game = training_run.game_class(1, training_run.agent)
    action = training_run.agent.get_actions(states, torch.tensor([True], dtype=torch.bool))

    graph = graph_from_mcts(training_run.agent.mcts)
    graph.draw(args.graph_path+'.2.png', prog='dot')

    print(training_run.game_class.action_to_str(action) if hasattr(training_run.game_class, 'action_to_str') else action)

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
