"""
Prints out a game between to naive MCTS players (dummy alpha evaluator)
"""
from aigames.game import CommandLineGame
from aigames.game.connect4 import Connect4
from aigames.agent.alpha_agent import AlphaAgent, DummyAlphaEvaluator, AlphaAgentHyperparameters


def main():
    alpha_evaluator = DummyAlphaEvaluator(7)
    hyperparams = AlphaAgentHyperparameters()
    hyperparams.n_mcts = 100
    alpha_agent = AlphaAgent(Connect4, alpha_evaluator, hyperparams)

    game = Connect4([alpha_agent, alpha_agent], listeners=[CommandLineGame()])
    game.play()


if __name__ == '__main__':
    main()
