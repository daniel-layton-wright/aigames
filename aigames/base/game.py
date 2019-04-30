import time
from typing import Callable, List

from aigames.base.agent import *


class Game:
    @staticmethod
    def reward(state, player_index):
        raise NotImplementedError()

    @staticmethod
    def legal_actions(state):
        raise NotImplementedError()

    @staticmethod
    def is_terminal_state(state):
        raise NotImplementedError()


class PartiallyObservableSequentialGame(Game):
    ILLEGAL_ACTION_PENALTY = 0

    def __init__(self, players: List[SequentialAgent], verbose=False, pause_seconds=0, monitor: Callable=None):
        self.players = players,
        self.verbose = verbose
        self.pause_seconds = pause_seconds
        self.monitor = monitor

    @staticmethod
    def get_next_state(state, action):
        raise NotImplementedError()

    @staticmethod
    def get_player_index(state):
        raise NotImplementedError()

    @staticmethod
    def get_observable_state(state, player_index):
        raise NotImplementedError()

    def play(self):
        for player in set(self.players):
            player.start_episode()

        while not self.is_terminal_state(self.state):
            i = self.get_player_index(self.state)
            player = self.players[i]
            # i, player correspond to the player about to move

            if self.verbose:
                print(self)
                time.sleep(self.pause_seconds)

            if self.monitor:
                self.monitor(self)

            observable_state = self.get_observable_state(self.state, i)
            next_action = player.choose_action(observable_state, i, verbose = self.verbose)

            while next_action not in self.legal_actions(observable_state):
                player.reward(self.ILLEGAL_ACTION_PENALTY, self.state, i)
                next_action = player.choose_action(observable_state, i, verbose=self.verbose)

            self.state = self.get_next_state(self.state, next_action)

            for j, p in enumerate(self.players):
                # j, p are just to issue rewards after each state
                p.reward(self.reward(self.state, j), self.state, j)

            if self.is_terminal_state(self.state):
                break

        if self.verbose:
            print(self)
            print('Game over.')

        for player in set(self.players):
            player.end_episode()


class SequentialGame(PartiallyObservableSequentialGame):
    @staticmethod
    def get_observable_state(state, i):
        # Fully observable
        return state
