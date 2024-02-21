from .game import GameListener
import os
import time
from .game_multi import GameListenerMulti


class CommandLineGame(GameListenerMulti):
    def __init__(self, pause_time=1, clear_screen=True, extra_prints=None):
        self.pause_time = pause_time
        self.clear_screen = clear_screen
        self.extra_prints = extra_prints if extra_prints is not None else []

    def before_action(self, game, legal_actions):
        if self.clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')
        print(str(game))

    def on_states_from_env(self, game):
        if self.clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')
        print(str(game))

    def after_action(self, game):
        if self.clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')
        print(str(game))

        for extra_print in self.extra_prints:
            print(str(extra_print))

        time.sleep(self.pause_time)

    def on_game_end(self, game):
        print('Game over.')
