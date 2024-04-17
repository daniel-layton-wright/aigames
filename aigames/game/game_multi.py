from collections import defaultdict
from typing import List, Tuple, Union, Set
import torch

from aigames.game import AbortGameException
from aigames.agent import AgentMulti, GameListenerMulti


class PauseGameException(Exception):
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
        """

        :return: next_states, rewards, is_env, is_terminal
        """
        raise NotImplementedError()

    @classmethod
    def get_env_action_idx(cls, states, legal_action_mask):
        raise NotImplementedError()

    @classmethod
    def get_next_states_from_env(self, states, action_idx=None) -> Tuple[torch.Tensor, torch.BoolTensor]:
        raise NotImplementedError()

    @classmethod
    def is_terminal(cls, states):
        raise NotImplementedError()

    @classmethod
    def get_legal_action_masks(cls, states):
        raise NotImplementedError()

    @classmethod
    def get_env_legal_action_masks(cls, states):
        raise NotImplementedError()

    def __init__(self, n_parallel_games, players: Union[AgentMulti, List[AgentMulti]], listeners: List[GameListenerMulti] = None):
        self.n_parallel_games = n_parallel_games
        self.states = self.get_initial_states(self.n_parallel_games)
        self.players = self.list_of_players(players)
        self.listeners = listeners if listeners is not None else []
        self.is_env = torch.zeros((self.n_parallel_games,), dtype=torch.bool, device=self.states.device)
        self.is_term = torch.ones((self.n_parallel_games,), dtype=torch.bool, device=self.states.device)

    def list_of_players(self, players: Union[AgentMulti, List[AgentMulti]]):
        if isinstance(players, list):
            assert(len(players) == self.get_n_players())
            return players
        else:
            return [players for _ in range(self.get_n_players())]

    @property
    def players_and_listeners(self) -> Set[GameListenerMulti]:
        return set(self.players) | set(self.listeners)

    def play(self):
        if self.is_term.all():
            # Restart the game from the initial state:
            self.states = self.get_initial_states(self.n_parallel_games)

            # Notify listeners game is starting:
            for player_or_listener in self.players_and_listeners:
                player_or_listener.before_game_start(self)

            self.is_env = torch.zeros((self.n_parallel_games,), dtype=torch.bool, device=self.states.device)
            self.is_term = torch.zeros((self.n_parallel_games,), dtype=torch.bool, device=self.states.device)
        else:
            # We are restarting a game mid-way through
            for player_or_listener in self.players_and_listeners:
                player_or_listener.before_game_start(self)

        try:
            while (~self.is_term).any():
                if self.is_env[~self.is_term].any():
                    env = ~self.is_term & self.is_env
                    for player_or_listener in self.players_and_listeners:
                        player_or_listener.before_env_move(self.states[env], env)

                    # Advance states that are in an env state
                    self.states[env], _, self.is_term[env] = self.get_next_states_from_env(self.states[env])
                    self.is_env[:] = False
                    continue

                for player_or_listener in self.players_and_listeners:
                    player_or_listener.on_states_from_env(self)

                # Ask the player for his move:
                going = ~self.is_term
                actions = self._get_actions()

                # Notify listeners of the move
                for player_or_listener in self.players_and_listeners:
                    player_or_listener.on_action(self, actions)

                # Advance to the next state according to the move
                self.states[going], rewards, self.is_env[going], self.is_term[going] = (
                    self.get_next_states(self.states[going], actions))

                for player_or_listener in self.players_and_listeners:
                    player_or_listener.on_rewards(rewards, going)

                for player_or_listener in self.players_and_listeners:
                    player_or_listener.after_action(self)

        except AbortGameException:
            self.is_term[:] = True
        except PauseGameException:
            return

        # Game over. Notify agents and listeners:
        for player_or_listener in self.players_and_listeners:
            player_or_listener.on_game_end(self)

    def _get_actions(self):
        # First we need to get the player index for each state
        going = ~self.is_term
        cur_player = self.get_cur_player_index(self.states[going])

        index_to_player = {i: player for i, player in enumerate(self.players)}

        # Now we need a list of (player, indices) for each player
        player_to_indices = defaultdict(lambda: torch.tensor([], dtype=torch.long))
        for i, player in enumerate(cur_player):
            j = index_to_player[player.item()]
            player_to_indices[j] = torch.cat((player_to_indices[j], torch.LongTensor([i])))

        actions = torch.zeros((going.sum(),), dtype=torch.long, device=self.states.device)
        for player, indices in player_to_indices.items():
            mask = going & torch.isin(torch.arange(going.shape[0]), torch.arange(going.shape[0])[going][indices])
            actions[indices] = player.get_actions(self.states[going][indices], mask)

        return actions

    def to(self, device):
        self.states = self.states.to(device)
        self.is_env = self.is_env.to(device)
        self.is_term = self.is_term.to(device)
        return self
