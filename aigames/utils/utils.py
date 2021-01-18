from .listeners import RewardListener


def play_tournament(game_class, players, n_games, discount_rate=0.99):
    reward_listener = RewardListener(discount_rate, 1)

    rewards = []
    for _ in range(n_games):
        game = game_class(players, [reward_listener])
        game.play()
        rewards.append( reward_listener.reward )

    return float(sum(rewards)) / len(rewards)