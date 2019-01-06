from agent import *
import numpy as np


class MinimaxAgent(SequentialAgent):
    def __init__(self, game):
        self.cache = {}
        self.game = game

    def reward(self, reward_value, state, i):
        pass

    def choose_action(self, state, agent_num, verbose = False):
        def V(s, r, level=0, verbose=False):
            state_key = MinimaxAgent.state_to_key(s)

            if state_key in self.cache:
                if verbose:
                    print(' ' * level + 'Found {} {} in cache: {}'.format(state_key, s, self.cache[state_key][2]))
                return self.cache[state_key]

            if verbose:
                print(' ' * level + 'Did not find {} {} in cache'.format(state_key, s))

            if self.game.is_terminal_state(s):
                if verbose:
                    print(' ' * level + 'This state is an end state. Utility {}'.format(self.game.reward(s, agent_num)))
                return (self.game.reward(s, agent_num), None, s)
            elif self.game.get_player_index(s) == agent_num:
                cur_opt = -1 * float('inf')
                cur_opt_actions = []
                for action in self.game.legal_actions(s):
                    new_state = self.game.get_next_state(s, action)
                    new_r = (max(r[0], cur_opt), r[1])
                    val, _, _ = V(new_state, new_r, level + 1, verbose)
                    if val > cur_opt:
                        cur_opt = val
                        cur_opt_actions = [action]
                    elif val == cur_opt:
                        cur_opt_actions.append(action)

                        # if cur_opt >= r[1]:
                        #	break

                self.cache[state_key] = (cur_opt, cur_opt_actions, s)
                return (cur_opt, cur_opt_actions, s)

            else:
                cur_opt = 1 * float('inf')
                cur_opt_actions = []
                for action in self.game.legal_actions(s):
                    new_state = self.game.get_next_state(s, action)
                    new_r = (r[0], min(cur_opt, r[1]))
                    val, _, _ = V(new_state, new_r, level + 1, verbose)
                    if val < cur_opt:
                        cur_opt = val
                        cur_opt_actions = [action]
                    elif val == cur_opt:
                        cur_opt_actions.append(action)

                        # if cur_opt <= r[0]:
                        #	break

                self.cache[state_key] = (cur_opt, cur_opt_actions, s)
                return (cur_opt, cur_opt_actions, s)

        _, actions, _ = V(state, (-1 * float('inf'), +1 * float('inf')))

        idx =  np.random.choice(len(actions))
        return actions[idx]

    def state_to_key(state):
        return tuple(state.flatten())