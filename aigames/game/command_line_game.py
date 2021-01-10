from .game import GameListener
import os
import time


class CommandLineGame(GameListener):
    def __init__(self, pause_time=1, clear_screen=True):
        self.pause_time = pause_time
        self.clear_screen = clear_screen

    def before_action(self, game, legal_actions):
        if self.clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')
        print(str(game))

    def after_action(self, game):
        if self.clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')
        print(str(game))
        time.sleep(self.pause_time)

    def on_game_end(self, game):
        print('Game over.')