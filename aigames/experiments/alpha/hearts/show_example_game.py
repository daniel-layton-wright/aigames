"""
Shows an example game of Hearts.
"""
from aigames.agent.hearts.simple_hearts_agent import SimpleHeartsAgent
from aigames.agent.random_agent_multi import RandomAgentMulti
from aigames.experiments.alpha.hearts.hearts import HeartsTrainingRun
from aigames.experiments.alpha.hearts.network_architectures import HeartsNetwork, HeartsNetworkHyperparameters
from aigames.game.hearts import Hearts
from aigames.game import CommandLineGame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--pause_time', type=float, default=5.)
    args = parser.parse_args()

    simple_hearts_agent = SimpleHeartsAgent()

    if args.ckpt_path is not None:
        training_run = HeartsTrainingRun.load_from_checkpoint(args.ckpt_path,
                                                              network=HeartsNetwork(HeartsNetworkHyperparameters()),
                                                              game_class=Hearts,
                                                              map_location='cpu')
        training_run.agent.eval()
        agents = [training_run.agent] + [simple_hearts_agent for _ in range(3)]
    else:
        agents = [simple_hearts_agent for _ in range(4)]

    cli = CommandLineGame(pause_time=args.pause_time)
    hearts = Hearts(1, agents, [cli])
    hearts.play()


if __name__ == '__main__':
    main()