import torch
from aigames.agent.alpha_agent_multi import DummyAlphaEvaluatorMulti, AlphaAgentHyperparametersMulti, AlphaAgentMulti
from aigames.utils.utils import add_all_slots_to_arg_parser, load_from_arg_parser
from aigames.mcts.mcts import MCTS


def test_mcts_speed(args, hypers):
    import os
    from aigames.game.G2048_multi import get_G2048Multi_game_class
    import perftester as pt

    if args.use_dummy_evaluator:
        evaluator = DummyAlphaEvaluatorMulti(4, 1, args.device)
    else:
        from aigames.experiments.alpha.G2048Multi.network_architectures import G2048MultiNetwork, G2048MultiEvaluator
        network = G2048MultiNetwork()
        network.eval()
        evaluator = G2048MultiEvaluator(network)

    G2048Multi = get_G2048Multi_game_class(args.device)
    agent = AlphaAgentMulti(G2048Multi, evaluator, hypers)
    game = G2048Multi(1, agent)

    # Fast version
    this_dir = os.path.dirname(os.path.abspath(__file__))
    fast_states = torch.load(os.path.join(this_dir, 'fast_state_for_mcts.pkl'), map_location=args.device)

    def run_fast_mcts():
        mcts = MCTS(game, evaluator, hypers, fast_states)
        mcts.search_for_n_iters(hypers.n_mcts_iters)

    slow_states = torch.load(os.path.join(this_dir, 'slow_state_for_mcts.pkl'), map_location=args.device)

    def run_slow_mcts():
        mcts = MCTS(game, evaluator, hypers, slow_states)
        mcts.search_for_n_iters(hypers.n_mcts_iters)

    run_fast_mcts()
    run_slow_mcts()

    fast_results = pt.time_benchmark(run_fast_mcts, Number=1, Repeat=10)
    slow_results = pt.time_benchmark(run_slow_mcts, Number=1, Repeat=10)

    print('Fast results: ')
    print(pt.pp(fast_results))

    print('Slow results: ')
    print(pt.pp(slow_results))


def main():
    hypers = AlphaAgentHyperparametersMulti()
    hypers.n_mcts_iters = 100

    import argparse
    parser = argparse.ArgumentParser()
    add_all_slots_to_arg_parser(parser, hypers)
    parser.add_argument('--use_dummy_evaluator', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    load_from_arg_parser(args, hypers)

    test_mcts_speed(args, hypers)


if __name__ == '__main__':
    main()


"""
Results local CPU:
Fast: avg 2.4s
Slow: avg 3.6s

When adding scaling to Q:
Fast: avg 2.6
Slow: avg 3.5

Trying to be clever on when to update min/maxQ (slower)
Fast: avg 3.3
Slow: avg 4.7
"""
