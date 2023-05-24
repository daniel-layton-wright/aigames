from .agent import Agent
from ..game import SequentialGame
import numpy as np
from typing import Type


class MinimaxAgent(Agent):
    def __init__(self, game: Type[SequentialGame], discount_rate=0.99):
        self.game = game
        self.cache = {}
        self.discount_rate = discount_rate

        if not hasattr(self.game, 'get_terminal_rewards'):
            raise ValueError('To use a MinimaxAgent, the game must have a get_terminal_rewards method')

    def on_reward(self, reward, next_state, player_index):
        pass

    def get_action(self, state, legal_actions) -> int:
        agent_num = self.game.get_cur_player_index(state)

        def V(s, r, level=0, verbose=False):
            state_key = MinimaxAgent.state_to_key(s)

            if state_key in self.cache:
                if verbose:
                    print(' ' * level + 'Found {} {} in cache: {}'.format(state_key, s, self.cache[state_key][2]))
                return self.cache[state_key]

            if verbose:
                print(' ' * level + 'Did not find {} {} in cache'.format(state_key, s))

            if self.game.is_terminal_state(s):
                return (0, None, s)

            elif self.game.get_cur_player_index(s) == agent_num:
                cur_opt = -1 * float('inf')
                cur_opt_actions = []
                for action in self.game.get_legal_actions(s):
                    new_state, rewards = self.game.get_next_state_and_rewards(s, action)
                    reward = rewards[agent_num]
                    new_r = (max(r[0], cur_opt), r[1])
                    val, _, _ = V(new_state, new_r, level + 1, verbose)
                    val = reward + self.discount_rate * val
                    if val > cur_opt:
                        cur_opt = val
                        cur_opt_actions = [action]
                    elif val == cur_opt:
                        cur_opt_actions.append(action)

                self.cache[state_key] = (cur_opt, cur_opt_actions, s)
                return (cur_opt, cur_opt_actions, s)

            else:
                cur_opt = 1 * float('inf')
                cur_opt_actions = []
                for action in self.game.get_legal_actions(s):
                    new_state, rewards = self.game.get_next_state_and_rewards(s, action)
                    reward = rewards[agent_num]
                    new_r = (r[0], min(cur_opt, r[1]))
                    val, _, _ = V(new_state, new_r, level + 1, verbose)
                    val = reward + self.discount_rate * val
                    if val < cur_opt:
                        cur_opt = val
                        cur_opt_actions = [action]
                    elif val == cur_opt:
                        cur_opt_actions.append(action)

                self.cache[state_key] = (cur_opt, cur_opt_actions, s)
                return (cur_opt, cur_opt_actions, s)

        _, actions, _ = V(state, (-1 * float('inf'), +1 * float('inf')))

        action_indices = [legal_actions.index(action) for action in actions]
        choice = np.random.choice(action_indices)
        return choice

    @staticmethod
    def state_to_key(state):
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        elif hasattr(state, 'hash'):
            return state.hash()
        else:
            raise ValueError('No way to convert state to key')
