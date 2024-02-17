"""
Prints out a game between to naive MCTS players (dummy alpha evaluator)
"""
from aigames.game import CommandLineGame
from aigames.game.connect4 import Connect4V2
from aigames.agent.alpha_agent import AlphaAgent, AlphaAgentHyperparameters
from aigames.experiments.alpha.connect4 import Connect4NetworkV2, Connect4EvaluatorV2


def main():
    network = Connect4NetworkV2()
    alpha_evaluator = Connect4EvaluatorV2(network)
    hyperparams = AlphaAgentHyperparameters()
    hyperparams.n_mcts = 0
    alpha_agent = AlphaAgent(Connect4V2, alpha_evaluator, hyperparams)

    network.eval()
    game = Connect4V2([alpha_agent, alpha_agent], listeners=[CommandLineGame(pause_time=1)])
    game.play()


if __name__ == '__main__':
    main()
