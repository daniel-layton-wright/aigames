from aigames.agent import ManualAgent
from aigames.game.twenty_forty_eight import TwentyFortyEight
from aigames.game import CommandLineGame


def main():
    cli = CommandLineGame()
    agents = [ManualAgent()]
    twenty_forty_eight = TwentyFortyEight(agents, [cli])
    twenty_forty_eight.play()


if __name__ == '__main__':
    main()