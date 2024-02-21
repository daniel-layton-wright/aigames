from ....game.G2048_multi import G2048Multi
from ....agent.g2048.simple_g2048_agent import SimpleG2048Agent
from ....utils.listeners import ActionCounterProgressBar


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=1000)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    agent = SimpleG2048Agent()
    game = G2048Multi(args.n_games, agent, listeners=[ActionCounterProgressBar(1000)])
    game.play()

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
