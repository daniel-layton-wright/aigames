from aigames.game.connect4 import *
from experiments.mp_alpha_connect4 import Connect4Network, Connect4Evaluator
from experiments.alpha_connect4 import *
from aigames.training_manager.alpha_training_manager import *
from aigames.agent.alpha_agent import *


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_mcts', type=int, default=1200)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    network = Connect4Network()
    network.load_state_dict(checkpoint['network_state'])

    evaluator = Connect4Evaluator(network)
    debug_listener = AlphaAgentDebugListener()
    agent = AlphaAgent(Connect4, evaluator, [debug_listener], use_tqdm=True, n_mcts=args.n_mcts)
    agent.eval()

    manual_agent = Connect4ClickAgent()
    listeners = [Connect4Gui(manual_agent)]

    game = Connect4([manual_agent, agent], listeners)
    game.play()



if __name__ == '__main__':
    main()