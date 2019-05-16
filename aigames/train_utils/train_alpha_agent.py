from aigames.agents import *


def train_alpha_agent(game_class, model: AlphaModel, monitor=None,
                      n_games=10000, alpha_agent_kwargs=dict()):
    alpha_agent = AlphaAgent(game_class, model, **alpha_agent_kwargs)
    monitor = monitor(model, alpha_agent)
    monitor.start()
    model.monitor = monitor
    players = [alpha_agent for _ in range(game_class.N_PLAYERS)]

    for i in range(n_games):
        cur_game = game_class(players)
        cur_game.play()
