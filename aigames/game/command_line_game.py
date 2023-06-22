from .game import GameListener
import os
import time
# TODO : do something about this


class CommandLineGame(GameListener):
    def __init__(self, pause_time=1, clear_screen=True):
        self.pause_time = pause_time
        self.clear_screen = clear_screen
        self.last_printed_lines = 0

    def before_action(self, game, legal_actions):
        if self.clear_screen:
            self.clear_n_lines(self.last_printed_lines)

        out = str(game)
        self.last_printed_lines = out.count('\n') + 1
        print(out)

    def after_action(self, game):
        if self.clear_screen:
            self.clear_n_lines(self.last_printed_lines)

        out = str(game)
        self.last_printed_lines = out.count('\n') + 1
        print(out)
        time.sleep(self.pause_time)

    def on_game_end(self, game):
        print('Game over.')

    @staticmethod
    def clear_n_lines(n):
        if n == 0:
            return

        print(f'\033[{n}A\033[J', end='')
