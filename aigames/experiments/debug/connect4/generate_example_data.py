from aigames.game.connect4 import *
from aigames.experiments import Connect4Network, Connect4Evaluator, Connect4Gui
from aigames.training_manager.alpha_training_manager import *
import pickle


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data.pkl')
    parser.add_argument('--n_games', type=int, default=2)
    parser.add_argument('--n_mcts', type=int, default=1200)
    parser.add_argument('--gui', dest='feature', action='store_true')
    parser.add_argument('--no-gui', dest='feature', action='store_false')
    parser.set_defaults(gui=True)
    args = parser.parse_args()

    network = Connect4Network()
    evaluator = Connect4Evaluator(network)
    dataset = BasicAlphaDataset(evaluator, process_state=True, min_size=1)
    listeners = []
    if args.gui:
        listeners.append(Connect4Gui())

    agent = AlphaAgent(Connect4, evaluator, [dataset], use_tqdm=True, n_mcts=args.n_mcts)
    game = Connect4([agent, agent], listeners)

    for _ in tqdm(range(args.n_games)):
        game.play()

    with open(args.file, 'wb') as f:
        pickle.dump(dataset, f)

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    main()
