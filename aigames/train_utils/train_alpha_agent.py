from aigames.agents import *


def train_alpha_agent(game_class, model: AlphaModel, optimizer_class, lr=0.01, monitor=None, model_device='cpu',
                      n_games=10000, alpha_agent_kwargs=dict()):
    optimizer = optimizer_class(model.parameters(), lr=lr)
    evaluator = AlphaEvaluator(model, model_device, optimizer)
    alpha_agent = AlphaAgent(game_class, evaluator, **alpha_agent_kwargs)
    monitor = monitor(model, alpha_agent)
    monitor.start()
    evaluator.monitor = monitor
    players = [alpha_agent for _ in range(game_class.N_PLAYERS)]

    for i in range(n_games):
        cur_game = game_class(players)
        cur_game.play()
