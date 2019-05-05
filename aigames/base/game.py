import time
from typing import Callable, List
import numpy as np
from aigames.base.agent import *


class Game:
    ALL_ACTIONS = []
    N_PLAYERS = 0

    @staticmethod
    def reward(state, player_index):
        raise NotImplementedError()

    @staticmethod
    def legal_actions(state):
        raise NotImplementedError()

    @staticmethod
    def is_terminal_state(state):
        raise NotImplementedError()

    @classmethod
    def legal_action_indices(cls, state):
        return np.array([cls.ALL_ACTIONS.index(action) for action in cls.legal_actions(state)])

    @classmethod
    def all_rewards(cls, state):
        return np.array([cls.reward(state, i) for i in range(cls.N_PLAYERS)])


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


class Tournament:
    def __init__(self, game_class, agents, n_games=100, game_kwargs=dict()):
        self.game_class = game_class
        self.agents = agents
        self.n_games = n_games
        self.game_kwargs = game_kwargs

    def play(self):
        final_states = []
        for i in range(self.n_games):
            cur_game = self.game_class(self.agents, **self.game_kwargs)
            cur_game.play()
            final_states.append(cur_game.state)

        return final_states
