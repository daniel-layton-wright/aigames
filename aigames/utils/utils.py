from typing import Optional, List
from .listeners import RewardListener, GameListener


def play_tournament(game_class, players, n_games, discount_rate=0.99, tqdm=True, listeners: Optional[List[GameListener]] = None):
    reward_listener = RewardListener(discount_rate, 1)
    listeners = listeners or []
    listeners.append(reward_listener)

    rewards = []
    if not tqdm:
        loop = range(n_games)
    else:
        from tqdm.auto import tqdm
        loop = tqdm(range(n_games), desc=f'{game_class.__name__} tournament')

    for _ in loop:
        game = game_class(players, listeners)
        game.play()
        rewards.append(reward_listener.reward)

    return float(sum(rewards)) / len(rewards)


def get_all_slots(obj):
    """
    For an object with a __slots__ variable which may inherit from other classes with __slots__ variables, this
    function collects all the slots into a single iterable.
    """
    from itertools import chain
    return chain.from_iterable(getattr(cls, '__slots__', []) for cls in type(obj).__mro__)


def add_all_slots_to_arg_parser(parser, obj):
    """
    For an object with a __slots__ variable which may inherit from other classes with __slots__ variables, this
    function adds all the slots to the given arg parser.

    :param parser: An argparse.ArgumentParser
    :param obj: An object with a __slots__ variables
    """
    for slot in get_all_slots(obj):
        parser.add_argument(f'--{slot}', type=type(getattr(obj, slot)), default=getattr(obj, slot),
                            help=f'The {slot} of {type(obj).__name__} class, type: {type(getattr(obj, slot)).__name__}')


def load_from_arg_parser(args, obj):
    """
    For an object with a __slots__ variable which may inherit from other classes with __slots__ variables, this
    function loads the slots from the given arg parser.

    :param args: An argparse.Namespace
    :param obj: An object with a __slots__ variable
    """
    for slot in get_all_slots(obj):
        setattr(obj, slot, getattr(args, slot))
