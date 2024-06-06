import numpy as np
import torch
from aigames.agent.alpha_agent_multi import DummyAlphaEvaluatorMulti, AlphaAgentHyperparametersMulti, AlphaAgentMulti, \
    ConstantMCTSIters, BaseAlphaEvaluator
from aigames.game.game_multi import GameMulti
from aigames.utils.utils import add_all_slots_to_arg_parser, load_from_arg_parser
from aigames.mcts.mcts import MCTS


def test_mcts_speed(args, hypers):
    import os
    from aigames.game.G2048_multi import get_G2048Multi_game_class
    import perftester as pt

    if args.use_dummy_evaluator:
        network = DummyAlphaEvaluatorMulti(4, 1, args.device)
    else:
        from aigames.experiments.alpha.G2048Multi.network_architectures import G2048MultiNetwork
        network = G2048MultiNetwork()
        network.eval()

    G2048Multi = get_G2048Multi_game_class(args.device)
    agent = AlphaAgentMulti(G2048Multi, network, hypers)
    game = G2048Multi(1, agent)

    # Fast version
    this_dir = os.path.dirname(os.path.abspath(__file__))
    fast_states = torch.load(os.path.join(this_dir, 'fast_state_for_mcts.pkl'), map_location=args.device).to(torch.uint8)

    def run_fast_mcts():
        n_mcts_iters = hypers.n_mcts_iters.get_n_mcts_iters()[0]
        mcts = MCTS(game, network, hypers, n_mcts_iters, fast_states)
        mcts.search_for_n_iters(n_mcts_iters)

    slow_states = torch.load(os.path.join(this_dir, 'slow_state_for_mcts.pkl'), map_location=args.device).to(torch.uint8)

    def run_slow_mcts():
        n_mcts_iters = hypers.n_mcts_iters.get_n_mcts_iters()[0]
        mcts = MCTS(game, network, hypers, n_mcts_iters, slow_states)
        mcts.search_for_n_iters(n_mcts_iters)

    test_mctx_speed(fast_states, game, network, hypers)
    run_fast_mcts()
    run_slow_mcts()

    fast_results = pt.time_benchmark(run_fast_mcts, Number=1, Repeat=10)
    slow_results = pt.time_benchmark(run_slow_mcts, Number=1, Repeat=10)

    mctx_results = pt.time_benchmark(lambda: test_mctx_speed(fast_states, game, network, hypers), Number=1, Repeat=10)

    print('Fast results: ')
    print(pt.pp(fast_results))

    print('Slow results: ')
    print(pt.pp(slow_results))

    print('mctx results: ')
    print(pt.pp(mctx_results))


def test_mctx_speed(initial_state, game: GameMulti, network: BaseAlphaEvaluator, hypers: AlphaAgentHyperparametersMulti):
    """
    This function uses GDM's mctx packaage to compare speeds
    """
    import mctx
    import jax.numpy as jnp

    batch_size = initial_state.shape[0]
    num_actions = game.get_n_actions()

    root_pi, root_value = network.evaluate(initial_state)

    # Convert pi distribution to logits
    prior_logits = jnp.log(jnp.array(root_pi))
    root_value = jnp.array(root_value.flatten())

    root = mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=root_value,
        # The embedding will hold the state index.
        embedding=jnp.array(initial_state),
    )

    root_pi = jnp.array(root_pi)

    def recurrent_fn(params, rng_key, action, embedding):
        """
        In the recurrent function we use the G2048Multi game to advance the state (embedding) and get the reward, etc.
        """
        # convert jax embedding to torch tensor
        def fwd(embedding, action):
            embedding = torch.from_numpy(np.asarray(embedding))
            action = torch.from_numpy(action)
            is_terminal = game.is_terminal(embedding)
            next_states = embedding
            rewards = torch.zeros((batch_size, 1))
            next_states[~is_terminal], rewards[~is_terminal], is_env, is_terminal[~is_terminal] = game.get_next_states(embedding[~is_terminal], action[~is_terminal])
            next_states[~is_terminal], is_terminal[~is_terminal] = game.get_next_states_from_env(next_states[~is_terminal])
            pi, value = network.evaluate(next_states)

            # mask pi by legal actions
            legal_action_masks = game.get_legal_action_masks(next_states)
            pi *= legal_action_masks

            return jnp.array(next_states), jnp.array(rewards.flatten()), jnp.array(pi), jnp.array(value.flatten())

        next_states, rewards, pi, value = jax.pure_callback(fwd,
                                        (jax.core.ShapedArray(embedding.shape, embedding.dtype),
                                         jax.core.ShapedArray(embedding.shape[:1], jnp.float32),
                                         jax.core.ShapedArray(root_pi.shape, root_pi.dtype),
                                         jax.core.ShapedArray(root_value.shape, root_value.dtype)), embedding, action)

        # Convert pi distribution to logits
        prior_logits = jnp.log(pi)
        value = jnp.array(value.flatten())

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=jnp.array(rewards.flatten()),
            discount=jnp.full((batch_size,), hypers.discount),
            prior_logits=prior_logits,
            value=value)

        return recurrent_fn_output, next_states

    # Do a search
    import mctx
    import jax

    # init rng_key
    rng_key = jax.random.PRNGKey(0)

    mctx.muzero_policy((), rng_key, root=root, recurrent_fn=recurrent_fn,
           num_simulations=hypers.n_mcts_iters.get_n_mcts_iters()[0],
           )


def main():
    hypers = AlphaAgentHyperparametersMulti()
    hypers.n_mcts_iters = ConstantMCTSIters(100)

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
