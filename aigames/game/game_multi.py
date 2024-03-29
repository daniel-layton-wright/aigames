from typing import List, Tuple
import torch
from aigames.game import AbortGameException, GameListener
from aigames.agent import AgentMulti


class GameListenerMulti(GameListener):
    def on_states_from_env(self, game):
        pass

    def on_rewards(self, rewards, mask):
        pass

    def on_game_restart(self, game):
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
    def get_state_dtype(cls) -> torch.dtype:
        return torch.float32

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
        self.is_env = torch.zeros((self.n_parallel_games,), dtype=torch.bool, device=self.states.device)
        self.is_term = torch.ones((self.n_parallel_games,), dtype=torch.bool, device=self.states.device)

    def play(self):
        if self.is_term.all():
            # Restart the game from the initial state:
            self.states = self.get_initial_states(self.n_parallel_games)

            # Notify listeners game is starting:
            for listener in self.listeners:
                listener.before_game_start(self)

            self.player.before_game_start(self)

            self.is_env = torch.zeros((self.n_parallel_games,), dtype=torch.bool, device=self.states.device)
            self.is_term = torch.zeros((self.n_parallel_games,), dtype=torch.bool, device=self.states.device)
        else:
            # We are restarting a game mid-way through
            for listener in self.listeners:
                listener.on_game_restart(self)

        try:
            while (~self.is_term).any():
                if self.is_env[~self.is_term].any():
                    self.player.before_env_move(self.states[~self.is_term & self.is_env], ~self.is_term & self.is_env)

                    # Advance states that are in an env state
                    self.states[~self.is_term & self.is_env], _, self.is_term[~self.is_term & self.is_env] = (
                        self.get_next_states_from_env(self.states[~self.is_term & self.is_env]))
                    self.is_env[:] = False
                    continue

                for listener in self.listeners:
                    listener.on_states_from_env(self)

                # Ask the player for his move:
                actions = self.player.get_actions(self.states[~self.is_term], ~self.is_term)

                # Notify listeners of the move
                for listener in self.listeners:
                    listener.on_action(self, actions)

                # Advance to the next state according to the move
                self.states[~self.is_term], rewards, self.is_env[~self.is_term], self.is_term[~self.is_term] = (
                    self.get_next_states(self.states[~self.is_term], actions))

                for listener in self.listeners:
                    listener.on_rewards(rewards, ~self.is_term)

                self.player.on_rewards(rewards, ~self.is_term)

                # Callback to listeners after the action
                for listener in self.listeners:
                    listener.after_action(self)

        except AbortGameException:
            self.is_term[:] = True

        # Game over. Notify agents and listeners:
        self.player.on_game_end()

        for listener in self.listeners:
            listener.on_game_end(self)

    def to(self, device):
        self.states = self.states.to(device)
        self.is_env = self.is_env.to(device)
        self.is_term = self.is_term.to(device)
        return self
