from .alpha_agent_multi import AlphaAgentMulti, TimestepData, AlphaAgentHyperparametersMulti


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

    def on_game_end(self):
        if not self.training:
            return

        trajectories = self.get_trajectories()
        num_moves_per_traj = [len(traj.states) for traj in trajectories]

        self.hyperparams.training_tau.update_metric('expected_num_moves', None)
        self.hyperparams.training_tau.update_metric('avg_total_num_moves',
                                                    sum(num_moves_per_traj) / float(len(num_moves_per_traj)))

        for data_listener in self.listeners:
            data_listener.on_trajectories(trajectories)
