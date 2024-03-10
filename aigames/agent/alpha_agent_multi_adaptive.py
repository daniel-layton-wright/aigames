from .alpha_agent_multi import AlphaAgentMulti, RewardData, TimestepData, AlphaAgentHyperparametersMulti, TDLambda
import torch


class AlphaAgentMultiAdaptive(AlphaAgentMulti):
    """
    An adaptive version of the AlphaAgentMulti. This agent will adjust the number of MCTS iterations and tau based on the
    expected number of moves left in the game. Which is predicted by the network and trained from the self play data
    """
    def __init__(self, game_class, network, hyperparams: AlphaAgentHyperparametersMulti, listeners=None):
        super().__init__(game_class, network, hyperparams, listeners)
        self.expected_num_moves = None

    def get_actions(self, states, mask):
        # Get expected num moves
        pi, v, self.expected_num_moves = self.evaluator.evaluate(states)
        self.hyperparams.training_tau.update_metric('expected_num_moves', self.expected_num_moves)
        return super().get_actions(states, mask)

    def record_pi(self, states, pi, mask, mcts_value=None, network_value=None, env_state=False):
        if not env_state:
            self.episode_history.append(TimestepData(states, pi, mask, mcts_value, network_value,
                                                     num_moves=self.expected_num_moves))
        else:
            super().record_pi(states, pi, mask, mcts_value, network_value, env_state)

    def generate_data_td_method(self, value_fn):
        """

        :param value_fn: A function that takes a TimestepData and returns the value to use for training. If the value is
        None, then the value is skipped in the TD summation

        """
        episode_history = reversed(self.episode_history)  # work backwards

        discounted_rewards_since_last_state = torch.zeros((self.game.n_parallel_games, self.game_class.get_n_players()),
                                                            dtype=torch.float32, device=self.game.states.device)

        last_state_vals = torch.zeros((self.game.n_parallel_games, self.game_class.get_n_players()),
                                      dtype=torch.float32, device=self.game.states.device)

        last_td_est = torch.zeros((self.game.n_parallel_games, self.game_class.get_n_players()),
                                  dtype=torch.float32, device=self.game.states.device)

        num_moves_left = torch.zeros((self.game.n_parallel_games, 1), dtype=torch.float32, device=self.game.states.device)
        num_moves = torch.zeros((self.game.n_parallel_games, 1), dtype=torch.float32, device=self.game.states.device)

        for data in episode_history:
            if isinstance(data, RewardData):
                full_rewards = torch.zeros((self.game.n_parallel_games, self.game_class.get_n_players()),
                                           dtype=torch.float32, device=data.reward_value.device)
                full_rewards[data.mask] = data.reward_value

                discounted_rewards_since_last_state = (full_rewards + self.hyperparams.discount *
                                                       discounted_rewards_since_last_state)
            elif isinstance(data, TimestepData):
                mask = data.mask

                if value_fn(data) is not None:
                    td_est = (discounted_rewards_since_last_state[mask]
                              + (1 - self.hyperparams.td_lambda.get_lambda()) * self.hyperparams.discount * last_state_vals[mask]
                              + self.hyperparams.td_lambda.get_lambda() * self.hyperparams.discount * last_td_est[mask])
                else:
                    td_est = last_td_est[mask]

                if data.num_moves is not None:
                    num_moves[mask] += 1
                    num_moves_left[mask] = (1
                                            + (1 - self.hyperparams.num_moves_td_lambda.get_lambda()) * data.num_moves
                                            + self.hyperparams.num_moves_td_lambda.get_lambda() * num_moves_left[mask])

                for data_listener in self.listeners:
                    data_listener.on_data_point(data.states, data.pis, td_est, num_moves_left[mask])

                if value_fn(data) is not None:
                    last_state_vals[mask] = value_fn(data)
                    discounted_rewards_since_last_state[mask] = 0
                    last_td_est[mask] = td_est

        self.hyperparams.training_tau.update_metric('avg_total_num_moves', num_moves.mean().item())
        self.hyperparams.training_tau.update_metric('expected_num_moves', None)
