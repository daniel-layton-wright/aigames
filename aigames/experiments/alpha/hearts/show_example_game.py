"""
Shows an example game of Hearts.
"""
from aigames.agent.hearts.simple_hearts_agent import SimpleHeartsAgent
from aigames.agent.random_agent_multi import RandomAgentMulti
from aigames.game.hearts import Hearts
from aigames.game import CommandLineGame


def main():
    cli = CommandLineGame()
    agent = SimpleHeartsAgent()
    agents = [agent for _ in range(4)]
    hearts = Hearts(1, agents, [cli])
    hearts.play()


if __name__ == '__main__':
    main()