from .game import GameListener
import os


class CommandLineGame(GameListener):
    def before_action(self, game, legal_actions):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(str(game))

    def after_action(self, game):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(str(game))

    def on_game_end(self, game):
        print('Game over.')