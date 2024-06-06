import argparse
import sys
from importlib import import_module
from typing import Optional, List
from .listeners import RewardListener, GameListener
from .. import GameListenerMulti


# TODO: replace this with more general version
def play_tournament_old(game_class, players, n_games, discount_rate=0.99, tqdm=True, listeners: Optional[List[GameListener]] = None):
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


def play_tournament(game_class, players, n_games, tqdm=True, listeners: Optional[List[GameListener]] = None):
    if not tqdm:
        loop = range(n_games)
    else:
        from tqdm.auto import tqdm
        loop = tqdm(range(n_games), desc=f'{game_class.__name__} tournament')

    for _ in loop:
        game = game_class(players, listeners)
        game.play()


def play_tournament_multi(game_class, players, n_parallel_games, n_rounds, tqdm=True,
                          listeners: Optional[List[GameListenerMulti]] = None):
    if not tqdm:
        loop = range(n_rounds)
    else:
        from tqdm.auto import tqdm
        loop = tqdm(range(n_rounds), desc=f'{game_class.__name__} tournament')

    for _ in loop:
        game = game_class(n_parallel_games, players, listeners)
        game.play()


def get_all_slots(obj):
    """
    For an object with a __slots__ variable which may inherit from other classes with __slots__ variables, this
    function collects all the slots into a single iterable.
    """
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return getattr(obj, '__slots__', [])
    else:
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
        if hasattr(obj, slot):
            cur_type = type(getattr(obj, slot))
            extra_args = {}

            if cur_type == bool:
                extra_args['action'] = argparse.BooleanOptionalAction

            parser.add_argument(f'--{slot}', type=cur_type, default=getattr(obj, slot),
                                help=f'The {slot} of {type(obj).__name__} class, type: {type(getattr(obj, slot)).__name__}',
                                **extra_args)


def load_from_arg_parser(args, obj):
    """
    For an object with a __slots__ variable which may inherit from other classes with __slots__ variables, this
    function loads the slots from the given arg parser.

    :param args: An argparse.Namespace
    :param obj: An object with a __slots__ variable
    """
    for slot in get_all_slots(obj):
        if hasattr(obj, slot):
            setattr(obj, slot, getattr(args, slot))


def cache(func):
    """
    A decorator for functions which will cache the result to file
    """
    try:
        import os
        import sys
        path = os.path.abspath(os.path.dirname(sys.modules[func.__module__].__file__))
        filename = f'{func.__module__}_{func.__name__}_cache.pkl'
        full_file_path = os.path.join(path, filename)

        import pickle

        if not os.path.exists(full_file_path):
            # Create the file and put in an empty dict
            with open(full_file_path, 'wb') as f:
                pickle.dump({}, f)

        def wrapped_func(*args, **kwargs):
            try:
                with open(full_file_path, 'rb') as f:
                    cache_dict = pickle.load(f)

                    # Hash args and kwargs
                    import hashlib
                    hash = hashlib.md5()
                    hash.update(pickle.dumps(args))
                    hash.update(pickle.dumps(kwargs))
                    hash = hash.hexdigest()

                    if hash in cache_dict:
                        return cache_dict[hash]
                    else:
                        result = func(*args, **kwargs)
                        cache_dict[hash] = result
                        with open(full_file_path, 'wb') as f:
                            pickle.dump(cache_dict, f)
                        return result
            except Exception as e:
                # If anything goes wrong just return the result from the original function and print warning message
                import warnings
                warnings.warn(f'Cache failed: {e}, using original function')
                return func(*args, **kwargs)

        return wrapped_func

    except Exception as e:
        # If anything goes wrong, just return the original function and make a warning message
        import warnings
        warnings.warn(f'Cache decorator failed: {e}, using original function')
        return func


def cached_import(module_path, class_name):
    # Check whether module is loaded and fully initialized.
    if not (
        (module := sys.modules.get(module_path))
        and (spec := getattr(module, "__spec__", None))
        and getattr(spec, "_initializing", False) is False
    ):
        module = import_module(module_path)
    return getattr(module, class_name)


def import_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as err:
        raise ImportError("%s doesn't look like a module path" % dotted_path) from err

    try:
        return cached_import(module_path, class_name)
    except AttributeError as err:
        raise ImportError(
            'Module "%s" does not define a "%s" attribute/class'
            % (module_path, class_name)
        ) from err


def bucketize(scaled_value, buckets, n_value_buckets):
    import torch
    bucketized = torch.clip(torch.bucketize(scaled_value, buckets), max=(n_value_buckets-1))
    bucketized_low = torch.clip(bucketized - 1, min=0)
    bucket_values = buckets[bucketized]
    one_less_buckets = buckets[bucketized_low]
    bucket_weight = torch.clip((scaled_value - one_less_buckets)
                               / (bucket_values - one_less_buckets + (bucket_values == one_less_buckets)),
                               min=0.0, max=1.0)
    out = torch.zeros((scaled_value.shape[0], n_value_buckets), device=scaled_value.device)
    out[torch.arange(scaled_value.shape[0]), bucketized.flatten()] = bucket_weight.flatten()
    out[torch.arange(scaled_value.shape[0]), bucketized_low.flatten()] = 1 - bucket_weight.flatten()
    return out


def digitize(bucketized_values, buckets):
    import torch
    return torch.sum(buckets * bucketized_values, dim=-1)
