from typing import List, Tuple
import torch
from aigames.game import AbortGameException, GameListener
from aigames.agent import AgentMulti


class GameListenerMulti(GameListener):
    def on_states_from_env(self, game):
        pass

    def on_rewards(self, rewards):
        pass


class GameMulti:
    @classmethod
    def get_n_players(cls) -> int:
        raise NotImplementedError()

    @classmethod
    def get_n_actions(cls) -> int:
        raise NotImplementedError()

    @classmethod
    def get_state_shape(cls) -> Tuple[int, ...]:
        raise NotImplementedError()

    @classmethod
    def get_n_stochastic_actions(cls) -> int:
        raise NotImplementedError()

    def get_initial_states(self, n) -> torch.Tensor:
        raise NotImplementedError()

    def get_cur_player_index(self, states) -> torch.Tensor:
        raise NotImplementedError()

    def get_next_states(self, states, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.BoolTensor]:
        raise NotImplementedError()

    def get_next_states_from_env(self, states) -> Tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
        raise NotImplementedError()

    @classmethod
    def is_terminal(cls, states):
        raise NotImplementedError()

    @classmethod
    def get_legal_action_masks(self, states):
        raise NotImplementedError()

    def __init__(self, n_parallel_games, player: AgentMulti, listeners: List[GameListenerMulti] = None):
        self.n_parallel_games = n_parallel_games
        self.states = self.get_initial_states(self.n_parallel_games)
        self.player = player  # Require that one agent be doing everything, otherwise it's sort of pointless
        self.listeners = listeners if listeners is not None else []

    def play(self):
        try:
            # Start the game from the initial state:
            self.states = self.get_initial_states(self.n_parallel_games)

            # Notify listeners game is starting:
            for listener in self.listeners:
                listener.before_game_start(self)

            self.player.before_game_start(self)

            is_env = torch.zeros((self.n_parallel_games,), dtype=torch.bool, device=self.states.device)
            is_terminal = torch.zeros((self.n_parallel_games,), dtype=torch.bool, device=self.states.device)

            while (~is_terminal).any():
                if is_env[~is_terminal].any():
                    self.player.before_env_move(self.states[~is_terminal], ~is_terminal)

                    # Advance states that are in an env state
                    self.states[~is_terminal & is_env], _, is_terminal[~is_terminal & is_env] = (
                        self.get_next_states_from_env(self.states[~is_terminal & is_env]))
                    is_env[:] = False
                    continue

                for listener in self.listeners:
                    listener.on_states_from_env(self)

                # Ask the player for his move:
                actions = self.player.get_actions(self.states[~is_terminal], ~is_terminal)

                # Notify listeners of the move
                for listener in self.listeners:
                    listener.on_action(self, actions)

                # Advance to the next state according to the move
                self.states[~is_terminal], rewards, is_env[~is_terminal], is_terminal[~is_terminal] = (
                    self.get_next_states(self.states[~is_terminal], actions))

                for listener in self.listeners:
                    listener.on_rewards(rewards)

                self.player.on_rewards(rewards, ~is_terminal)

                # Callback to listeners after the action
                for listener in self.listeners:
                    listener.after_action(self)

            # Game over. Notify agents and listeners:
            self.player.on_game_end()

            for listener in self.listeners:
                listener.on_game_end(self)

        except AbortGameException:
            return
