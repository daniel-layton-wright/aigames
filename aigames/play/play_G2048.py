from aigames.agent import ManualAgent
from aigames.game.G2048 import G2048
from aigames.game import CommandLineGame


def main():
    cli = CommandLineGame()
    agents = [ManualAgent()]
    twenty_forty_eight = G2048(agents, [cli])
    twenty_forty_eight.play()


if __name__ == '__main__':
    main()