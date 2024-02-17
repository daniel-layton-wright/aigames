import torch
import os
import sys
top_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))
sys.path.insert(0, top_dir)
from aigames import *
from aigames.experiments import TicTacToeAlphaModel


class DebuggingMonitor:
    def on_choose_action(self, agent, state, action):
        print(agent.cur_node.state)
        print(agent.cur_node.children_N)
        print(agent.cur_node.children_Q)
        print(agent.cur_node.v)
        print(agent.cur_node.P)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str)
    args = parser.parse_args()

    model_state_dict = torch.load(args.checkpoint, map_location='cpu')
    model = TicTacToeAlphaModel(TicTacToe)
    model.load_state_dict(model_state_dict)

    monitor = DebuggingMonitor()
    agent = AlphaAgent(TicTacToe, model, monitor=monitor)
    agent.eval()

    minimax_agent = MinimaxAgent(TicTacToe)

    game = TicTacToe([minimax_agent, agent], verbose=True)
    game.play()

if __name__ == '__main__':
    main()