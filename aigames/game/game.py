from aigames.agent import Agent
from typing import List
import warnings


class GameListener:
    def before_game_start(self, game):
        pass

    def before_action(self, game, legal_actions):
        pass

    def on_action(self, game, action):
        pass

    def after_action(self, game):
        pass

    def on_game_end(self, game):
        pass


class AbortGameException(Exception):
    pass


class PartiallyObservableSequentialGame:
    def __init__(self, players: List[Agent], listeners: List[GameListener] = None):
        self.state = self.get_initial_state()
        self.players = players
        self.listeners = listeners if listeners is not None else []

    def play(self):
        try:
            # Start the game from the initial state:
            self.state = self.get_initial_state()

            # Notify listeners game is starting:
            for listener in self.listeners:
                listener.before_game_start(self)

            for player in self.players:
                player.before_game_start(n_players=len(self.players))

            while not self.is_terminal_state(self.state):

                # Ask the player for his move:
                cur_player_index = self.get_cur_player_index(self.state)
                cur_player = self.players[cur_player_index]
                legal_actions = self.get_legal_actions(self.state)

                if len(legal_actions) == 0:
                    warnings.warn('There are no legal actions. This should not happen. Ending the game.')
                    break

                for listener in self.listeners:
                    listener.before_action(self, legal_actions)

                observable_state = self.get_observable_state(self.state, cur_player_index)

                cur_action_index = cur_player.get_action(observable_state, legal_actions)

                cur_action = legal_actions[cur_action_index]

                # Notify listeners of the move
                for listener in self.listeners:
                    listener.on_action(self, cur_action)

                # Advance to the next state according to the move
                self.state, rewards = self.get_next_state_and_rewards(self.state, cur_action)

                # Tell each player what his reward is in the next state
                for player_index, (player, reward) in enumerate(zip(self.players, rewards)):
                    player.on_reward(reward, self.get_observable_state(self.state, player_index), player_index)

                # Callback to listeners after the action
                for listener in self.listeners:
                    listener.after_action(self)

            # Game over. Notify agents and listeners:
            for player in self.players:
                player.on_game_end()

            for listener in self.listeners:
                listener.on_game_end(self)

        except AbortGameException:
            return

    @classmethod
    def is_terminal_state(cls, state):
        raise NotImplementedError()

    @classmethod
    def get_observable_state(cls, state, player_index):
        raise NotImplementedError()

    @classmethod
    def get_cur_player_index(cls, state) -> int:
        raise NotImplementedError()

    @classmethod
    def get_next_state_and_rewards(cls, state, action):
        raise NotImplementedError()

    @classmethod
    def get_rewards(cls, state):
        raise NotImplementedError()

    @classmethod
    def get_initial_state(cls):
        raise NotImplementedError()
    
    @classmethod
    def get_all_actions(cls) -> List:
        raise NotImplementedError()

    @classmethod
    def get_legal_actions(cls, state) -> List:
        raise NotImplementedError()

    @classmethod
    def get_n_players(cls):
        raise NotImplementedError()

    @classmethod
    def states_equal(cls, state1, state2):
        raise NotImplementedError()


class SequentialGame(PartiallyObservableSequentialGame):
    def get_observable_state(cls, state, player_index):
        return state
