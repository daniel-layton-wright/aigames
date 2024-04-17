"""
Prints out a game between to naive MCTS players (dummy alpha evaluator)
"""
from ....game import CommandLineGame
from ....game.connect4_multi import Connect4Multi
from ....agent.alpha_agent_multi import AlphaAgentMulti, \
    AlphaAgentHyperparametersMulti as AlphaAgentHyperparametersMulti, DummyAlphaEvaluatorMulti, ConstantMCTSIters
from ....utils.listeners import AvgRewardListenerMulti, ActionCounterProgressBar
from ....utils.utils import add_all_slots_to_arg_parser, load_from_arg_parser
import argparse
from .network_architectures import Connect4Network
import torch


def main():
    hyperparams = AlphaAgentHyperparametersMulti()
    hyperparams.n_mcts_iters = ConstantMCTSIters(100)

    # Set up an argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use_network', action='store_true')
    parser.add_argument('--n_games', type=int, default=1)
    parser.add_argument('--hide_game', action='store_true')
    parser.add_argument('--show_avg_score', action='store_true')
    parser.add_argument('--show_action_counter', action='store_true')
    parser.add_argument('--pdb', action='store_true')
    parser.add_argument('--pickle_history_path', type=str, default=None)

    add_all_slots_to_arg_parser(parser, hyperparams)

    # Parse the args
    args = parser.parse_args()

    load_from_arg_parser(args, hyperparams)

    # If using a network set up the network, otherwise use a dummy evaluator
    if args.use_network:
        alpha_evaluator = Connect4Network()
        alpha_evaluator.forward = torch.jit.trace(alpha_evaluator.forward, torch.rand((2, 1, 6, 7)))
        alpha_evaluator.eval()
    else:
        alpha_evaluator = DummyAlphaEvaluatorMulti(7, 2, args.device)

    alpha_agent = AlphaAgentMulti(Connect4Multi, alpha_evaluator, hyperparams)

    listeners = []
    if not args.hide_game:
        listeners = [CommandLineGame(pause_time=0.1)]

    if args.show_avg_score:
        listeners.append(AvgRewardListenerMulti(hyperparams.discount, 0, show_tqdm=True, tqdm_total=20000))

    if args.show_action_counter:
        listeners.append(ActionCounterProgressBar(42))

    game = Connect4Multi(args.n_games, alpha_agent, listeners)
    game.play()

    if args.pickle_history_path is not None:
        # Pickle the agent object to the given path
        import pickle
        with open(args.pickle_history_path, 'wb') as f:
            pickle.dump({'hyperparams': hyperparams, 'episode_history': alpha_agent.episode_history}, f)

    if args.pdb:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
