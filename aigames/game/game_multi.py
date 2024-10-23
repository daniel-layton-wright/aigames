from collections import defaultdict
from typing import List, Tuple, Union, Set
import torch

from aigames.game import AbortGameException
from aigames.agent import AgentMulti, GameListenerMulti


class PauseGameException(Exception):
    """Exception to pause the game."""
    pass


class PartiallyObservableGameMulti:
    """
    Base class for partially observable multi-player games.

    This class handles the core mechanisms related to partial observability,
    including state initialization, environment state handling, and notifying listeners.
    """

    @classmethod
    def get_n_players(cls) -> int:
        """Get the number of players in the game."""
        raise NotImplementedError()

    @classmethod
    def get_n_actions(cls) -> int:
        """Get the number of possible actions in the game."""
        raise NotImplementedError()

    @classmethod
    def get_state_shape(cls) -> Tuple[int, ...]:
        """Get the shape of the game state tensor."""
        raise NotImplementedError()

    @classmethod
    def get_state_dtype(cls) -> torch.dtype:
        """Get the data type of the game state tensor."""
        return torch.float32

    @classmethod
    def get_n_stochastic_actions(cls) -> int:
        """Get the number of stochastic actions."""
        raise NotImplementedError()

    def get_initial_states(self, n: int) -> torch.Tensor:
        """
        Initialize and return the initial states for n parallel games.

        :param n: Number of parallel games.
        :return: Tensor of initial states.
        """
        raise NotImplementedError()

    def get_cur_player_index(self, states: torch.Tensor) -> torch.Tensor:
        """
        Determine the current player's index for each game state.

        :param states: Tensor of current states.
        :return: Tensor of current player indices.
        """
        raise NotImplementedError()

    def get_next_states(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.BoolTensor]:
        """
        Compute the next states, rewards, environment flags, and terminal flags given current states and actions.

        :param states: Current states tensor.
        :param actions: Actions taken.
        :return: A tuple containing next_states, rewards, is_env, is_terminal.
        """
        raise NotImplementedError()

    @classmethod
    def get_env_action_idx(cls, states: torch.Tensor, legal_action_mask: torch.Tensor) -> torch.Tensor:
        """Get the environment action indices based on the states and legal action masks."""
        raise NotImplementedError()

    @classmethod
    def get_next_states_from_env(
        cls, 
        states: torch.Tensor, 
        action_idx: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Advance the state based on environment actions.

        :param states: Current states tensor.
        :param action_idx: Optional action indices.
        :return: A tuple containing next_states and is_terminal flags.
        """
        raise NotImplementedError()

    @classmethod
    def is_terminal(cls, states: torch.Tensor) -> torch.BoolTensor:
        """Determine if the given states are terminal."""
        raise NotImplementedError()

    @classmethod
    def get_legal_action_masks(cls, states: torch.Tensor) -> torch.Tensor:
        """Get masks indicating legal actions for the given states."""
        raise NotImplementedError()

    @classmethod
    def get_env_legal_action_masks(cls, states: torch.Tensor) -> torch.Tensor:
        """Get masks for legal environment actions."""
        raise NotImplementedError()

    def get_observable_states(
        self, 
        states: torch.Tensor, 
        player_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the observable states from the perspective of specific players.

        :param states: Current states tensor.
        :param player_indices: Tensor of player indices for observability.
        :return: Tensor of observable states.
        """
        raise NotImplementedError()

    def __init__(
        self,
        n_parallel_games: int,
        players: Union[AgentMulti, List[AgentMulti]],
        listeners: List[GameListenerMulti] = None
    ):
        """
        Initialize the multi-player game.

        :param n_parallel_games: Number of parallel game instances.
        :param players: A single AgentMulti instance or a list of AgentMulti instances.
        :param listeners: Optional list of GameListenerMulti instances.
        """
        self.n_parallel_games = n_parallel_games
        self.states = self.get_initial_states(self.n_parallel_games)
        self.players = self.list_of_players(players)
        self.listeners = listeners if listeners is not None else []
        self.is_env = torch.zeros(
            (self.n_parallel_games,), dtype=torch.bool, device=self.states.device
        )
        self.is_term = torch.ones(
            (self.n_parallel_games,), dtype=torch.bool, device=self.states.device
        )

    def list_of_players(
        self,
        players: Union[AgentMulti, List[AgentMulti]]
    ) -> List[AgentMulti]:
        """
        Ensure that players are in a list format.

        :param players: A single AgentMulti instance or a list of AgentMulti instances.
        :return: List of AgentMulti instances.
        """
        if isinstance(players, list):
            assert len(players) == self.get_n_players(), "Number of players does not match."
            return players
        else:
            return [players for _ in range(self.get_n_players())]

    @property
    def players_and_listeners(self) -> Set[GameListenerMulti]:
        """Get a set containing all players and listeners."""
        return set(self.players) | set(self.listeners)

    @property
    def player(self) -> AgentMulti:
        """
        Get the single player if all players are the same.

        :raises ValueError: If not all players are the same.
        :return: The single AgentMulti instance.
        """
        if len(set(self.players)) == 1:
            return self.players[0]
        else:
            raise ValueError(
                "All players must be the same to use the GameMulti.player attribute"
            )

    @player.setter
    def player(self, player: AgentMulti) -> None:
        """
        Set the player for all positions in the game.

        :param player: The AgentMulti instance to set as the player.
        """
        self.players = self.list_of_players(player)

    def play(self) -> None:
        """
        Play the game until all parallel instances reach terminal states.

        Handles environment moves, player actions, and notifies listeners accordingly.
        """
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
                    # Notify environment with full states
                    for player_or_listener in self.players_and_listeners:
                        player_or_listener.before_env_move(self.states[env], env)

                    # Advance states that are in an env state
                    self.states[env], self.is_term[env] = self.get_next_states_from_env(
                        self.states[env]
                    )
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
                (
                    self.states[going],
                    rewards,
                    self.is_env[going],
                    self.is_term[going],
                ) = self.get_next_states(self.states[going], actions)

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

    def _get_actions(self) -> torch.Tensor:
        """
        Retrieve actions for the current non-terminal game states.

        :return: Tensor of actions.
        """
        # Identify the games that are still in progress
        going = (~self.is_term).cpu()
        cur_player = self.get_cur_player_index(self.states[going])

        index_to_player = {i: player for i, player in enumerate(self.players)}

        # Now we need a list of (player, indices) for each player
        player_to_indices = defaultdict(lambda: torch.tensor([], dtype=torch.long))
        for i, player in enumerate(cur_player):
            j = index_to_player[player.item()]
            player_to_indices[j] = torch.cat((player_to_indices[j], torch.LongTensor([i])))

        actions = torch.zeros((going.sum(),), dtype=torch.long, device=self.states.device)
        for player, indices in player_to_indices.items():
            observable_states = self.get_observable_states(
                self.states[going][indices], 
                cur_player[indices]
            )
            mask = going & torch.isin(
                torch.arange(going.shape[0]),
                torch.arange(going.shape[0])[going][indices],
            )
            actions[indices] = player.get_actions(observable_states, mask)

        return actions

    def to(self, device: torch.device) -> 'PartiallyObservableGameMulti':
        """
        Move game states and flags to the specified device.

        :param device: The target device.
        :return: Self for chaining.
        """
        self.states = self.states.to(device)
        self.is_env = self.is_env.to(device)
        self.is_term = self.is_term.to(device)
        return self


class GameMulti(PartiallyObservableGameMulti):
    """
    Multi-player game class inheriting from PartiallyObservableGameMulti.

    This class retains all functionalities of GameMulti while leveraging partial observability features.
    In this fully observable variant, observable states are equal to the actual states.
    """

    @classmethod
    def get_observable_states(
        cls, 
        states: torch.Tensor, 
        player_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the actual states as observable states since the game is fully observable.

        :param states: Current states tensor.
        :param player_indices: Tensor of player indices for observability.
        :return: Tensor of observable states (same as actual states).
        """
        return states
