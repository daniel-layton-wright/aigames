from aigames.training_manager.alpha_training_manager import *
from aigames.agent.minimax_agent import *
from alphatoe import *


class DebugAlphaAgentListener(AlphaAgentListener):
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def after_mcts_search(self, root_node):
        print(root_node.children_total_N)

    def on_data_point(self, state, pi, reward):
        print(self.evaluator.process_state(state), pi, reward)


def main():
    network = TicTacToeNetwork()
    evaluator = FastTicTacToeEvaluator(network)
    agent = AlphaAgent(FastTicTacToe, evaluator, listeners=[DebugAlphaAgentListener(evaluator)])
    cli = CommandLineGame(clear_screen=False)
    manual_agent = ManualAgent()
    minimax_agent = MinimaxAgent(FastTicTacToe)
    game = FastTicTacToe([minimax_agent, agent], [cli])
    game.play()
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    main()