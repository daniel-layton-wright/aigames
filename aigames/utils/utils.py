from .listeners import RewardListener


def play_tournament(game_class, players, n_games, discount_rate=0.99):
    reward_listener = RewardListener(discount_rate, 1)

    rewards = []
    for _ in range(n_games):
        game = game_class(players, [reward_listener])
        game.play()
        rewards.append( reward_listener.reward )

    return float(sum(rewards)) / len(rewards)


def get_all_slots(obj):
    """
    For an object with a __slots__ variable which may inherit from other classes with __slots__ variables, this
    function collects all the slots into a single iterable.
    """
    from itertools import chain
    return chain.from_iterable(getattr(cls, '__slots__', []) for cls in type(obj).__mro__)
