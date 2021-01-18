from .agent import Agent
import numpy as np
from ..game import PartiallyObservableSequentialGame
from typing import Type
import copy


class QLearningDataListener:
    def on_SARS(self, state, action_index, reward, next_state):
        raise NotImplementedError()


class QLearningFunction:
    def evaluate(self, state):
        raise NotImplementedError()


class ExplorationProbabilityScheduler:
    def get_exploration_probability(self, agent, state):
        raise NotImplementedError()


class QLearningAgent(Agent):
    def __init__(self, game: Type[PartiallyObservableSequentialGame], player_index,
                 Q: QLearningFunction, *,  # After this, everything must be passed as a keyword arg
                 data_listener: QLearningDataListener,
                 exploration_probability_scheduler: ExplorationProbabilityScheduler, discount_rate):
        self.game = game
        self.all_actions = game.get_all_actions()
        self.player_index = player_index
        self.Q = Q
        self.data_listener = data_listener
        self.exploration_probability_scheduler = exploration_probability_scheduler
        self.discount_rate = discount_rate
        self.last_state = None
        self.last_action_index = None
        self.cum_reward = 0

        self.training = True

    def get_action(self, state, legal_actions) -> int:
        self.last_state = copy.deepcopy(state)
        legal_action_indices = [self.all_actions.index(legal_action) for legal_action in legal_actions]

        if self.training:
            exploration_probability = self.exploration_probability_scheduler.get_exploration_probability(self, state)
        else:
            exploration_probability = 0

        if np.random.random() < exploration_probability:
            # Explore randomly
            action_idx = np.random.randint(len(legal_actions))
        else:
            # Exploit: evaluate the network on the state and take the max
            scores = self.Q.evaluate(state)

            # The scores will be for all actions, but we need to get the best *legal* action:
            action_idx = scores[legal_action_indices].max(0)[1].item()

        self.last_action_index = legal_action_indices[action_idx]
        return action_idx

    def on_reward(self, reward, next_state, player_index):
        if self.game.get_cur_player_index(next_state) == self.player_index or self.game.is_terminal_state(next_state):
            # It's back to this player's turn (or it's the end of the game)
            self.cum_reward += reward

            if self.last_state is not None and self.training:
                # Call back to the data listener on a data point (if we are training)
                self.data_listener.on_SARS(self.last_state, self.last_action_index, self.cum_reward, copy.deepcopy(next_state))

            # Reset the cum_reward
            self.cum_reward = 0
        else:
            # It's someone else's turn let's just keep track of the reward
            self.cum_reward += reward

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def before_game_start(self, n_players):
        self.last_state = None
