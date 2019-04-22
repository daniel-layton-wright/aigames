from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import os
import numpy as np
import torch


class AlphaTTTMonitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.n_games = 0
        self.logger = SummaryWriter(log_dir)

        self.test_state = np.array([
            [[1, 0, 1],
             [0, 0, 0],
             [0, 0, 0]],
            [[0, 1, 0],
             [0, 0, 0],
             [0, 1, 0]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
        ])

    def debug(self, alpha_training_run, loss):
        self.logger.add_scalar('loss', loss, self.n_games)

        p, v = alpha_training_run.evaluator(self.test_state)
        fig, ax = plt.subplots()
        ax.bar(range(len(p)), p)
        ax.set_xticks(range(len(p)))
        ax.set_xticklabels(alpha_training_run.game_class.ALL_ACTIONS)
        self.logger.add_figure('test state actions', fig, self.n_games)
        plt.close(fig)

        state_str = alpha_training_run.game_class.state_to_str(self.test_state)
        self.logger.add_text('Test State', state_str, self.n_games)

        legal_actions = alpha_training_run.game_class.legal_actions(self.test_state)
        self.logger.add_text('Legal Action', str(legal_actions), self.n_games)

        if (self.n_games + 1) % 100 == 0:
            pct_losses = alpha_training_run.evaluate()
            self.logger.add_scalar('pct_losses', pct_losses, self.n_games)

        self.n_games += 1


class AlphaConnect4Monitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.n_games = 0
        self.logger = SummaryWriter(log_dir)
        self.manual_agent = ManualAgent()

    def debug(self, alpha_training_run, game, loss):
        self.logger.add_scalar('loss', loss, self.n_games)
        self.logger.add_scalar('game_length', game.state[:2,:,:].sum(), self.n_games)
        self.logger.add_scalar('first_player_win', int(alpha_training_run.game_class.reward(game.state, 0) == 1), self.n_games)

        if (self.n_games + 1) % 100 == 0:
            with open(os.path.join(self.log_dir, 'connect4.pt'), 'wb') as f:
                torch.save(alpha_training_run.agent, f)

        self.n_games += 1