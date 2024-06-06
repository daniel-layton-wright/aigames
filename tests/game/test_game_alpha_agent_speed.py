from typing import Type, Union
from aigames.game.game_multi import GameMulti
from aigames.mcts.mcts import MCTS
from aigames.utils.listeners import ActionCounterProgressBar, MaxActionGameKiller
from aigames.utils.utils import import_string
import time


def test_alpha_agent_playing_speed(game_class: Type[GameMulti], n_parallel_games: int = 100,
                                   max_n_actions: Union[int, None] = None, do_without_reuse_mcts=True,
                                   do_with_reuse_mcts=True, debug=False):
    """
    Test the speed of the AlphaAgent playing a game

    :param game_class: The game class to play
    :param n_parallel_games: The number of parallel games to play
    :param max_n_actions: The maximum number of actions to play before killing the game
    :param do_without_reuse_mcts: Whether to do the test without reuse_mcts_tree
    :param do_with_reuse_mcts: Whether to do the test with reuse_mcts_tree

    """
    from tests.helpers import Timer
    # Play a game of hearts with an alpha agent and record the time taken
    from aigames.agent.alpha_agent_multi import AlphaAgentMulti, DummyAlphaEvaluatorMulti, AlphaAgentHyperparametersMulti

    listeners = []
    if max_n_actions is not None:
        listeners.append(MaxActionGameKiller(max_n_actions))
        listeners.append(ActionCounterProgressBar(max_n_actions))

    eval = DummyAlphaEvaluatorMulti(game_class.get_n_actions(), game_class.get_n_players())

    # Make a wrapper decorator to keep track of the number of calls to evaluate
    class Counter:
        def __init__(self):
            self.count = 0

        def decorate(self, f):
            def wrapper(*args, **kwargs):
                self.count += 1
                return f(*args, **kwargs)
            return wrapper

    counter = Counter()
    eval.evaluate = counter.decorate(eval.evaluate)

    # Make a wrapper decorator to track the time spend doing get_next_mcts
    class CumulativeTimer:
        def __init__(self):
            self.duration = 0

        def decorate(self, f):
            def wrapper(*args, **kwargs):
                t = Timer()
                with t:
                    result = f(*args, **kwargs)
                self.duration += t.duration_in_seconds()
                return result
            return wrapper

    cumulative_timer = CumulativeTimer()
    MCTS.get_next_mcts = cumulative_timer.decorate(MCTS.get_next_mcts)

    hypers = AlphaAgentHyperparametersMulti()
    hypers.expand_simultaneous_fraction = 1.0
    alpha_agent = AlphaAgentMulti(game_class, eval, hypers)
    game = game_class(n_parallel_games, alpha_agent, listeners=listeners)

    if do_without_reuse_mcts:
        t = Timer()
        with t:
            game.play()

        print(f'Time for AlphaAgent playing {game_class.__name__}: {t.duration_in_seconds()} seconds')
        print(f'Number of calls to evaluate: {counter.count}')
        print(f'Time spent in get_next_mcts: {cumulative_timer.duration} seconds')

    if do_with_reuse_mcts:
        hypers.reuse_mcts_tree = True
        counter.count = 0  # reset count
        cumulative_timer.duration = 0  # reset duration

        t = Timer()
        with t:
            game.play()

        print(f'Time for AlphaAgent playing {game_class.__name__} with reuse_mcts_tree: {t.duration_in_seconds()} seconds')
        print(f'Number of calls to evaluate: {counter.count}')
        print(f'Time spent in get_next_mcts: {cumulative_timer.duration} seconds')

    if debug:
        import pdb
        pdb.set_trace()


def main():
    # Setup arg parser for game class and n parallel games
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_class', type=str, required=True)
    parser.add_argument('--n_parallel_games', type=int, default=100)
    parser.add_argument('--max_n_actions', type=int, default=None)
    parser.add_argument('--do_without_reuse_mcts', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--do_with_reuse_mcts', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # Get the game class
    game_class = import_string(args.game_class)

    # Run the test
    test_alpha_agent_playing_speed(game_class, args.n_parallel_games, args.max_n_actions, args.do_without_reuse_mcts,
                                   args.do_with_reuse_mcts, args.debug)


if __name__ == '__main__':
    main()
