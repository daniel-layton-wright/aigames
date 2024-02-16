import argparse
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from aigames import CommandLineGame
from aigames.agent.alpha_agent import TrainingTau
from aigames.agent.alpha_agent_multi import AlphaAgentMultiListener
from aigames.training_manager.alpha_training_manager_multi_lightning import AlphaMultiTrainingRunLightning, \
    AlphaMultiTrainingHyperparameters
from aigames.utils.listeners import ActionCounterProgressBar
from .network_architectures import G2048MultiNetwork, G2048MultiEvaluator
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from aigames.utils.utils import get_all_slots, add_all_slots_to_arg_parser, load_from_arg_parser
import os
import torch
from aigames.game.G2048_multi import get_G2048Multi_game_class


class NetworkMCTSMonitor(AlphaAgentMultiListener):
    def __init__(self):
        super().__init__()
        self.pi = None
        self.chosen_actions = None
        self.network_pi = None

    def after_mcts_search(self, mcts, pi, chosen_actions):
        self.pi = pi
        self.chosen_actions = chosen_actions
        self.network_pi = mcts.pi[:, 1].cpu().numpy()
        self.network_value = mcts.values[:, 1].cpu().numpy()

    def __str__(self):
        # Print out the pi, the mcts search results, the chsen actions
        out = ''
        out += f'network pi: {self.network_pi}\n'
        out += f'network value: {self.network_value}\n'
        out += f'mcts results: {self.pi}\n'
        out += f'chosen_actions: {self.chosen_actions}\n'
        return out


def main():
    network = G2048MultiNetwork(n_blocks=2, n_channels=64, n_out_channels=16)
    network_mcts_monitor = NetworkMCTSMonitor()

    hyperparams = AlphaMultiTrainingHyperparameters()
    hyperparams.self_play_every_n_epochs = 10
    hyperparams.n_parallel_games = 1
    hyperparams.max_data_size = 200000
    hyperparams.min_data_size = 512
    hyperparams.n_mcts_iters = 250
    hyperparams.dirichlet_alpha = 1.0
    hyperparams.dirichlet_epsilon = 0.25
    hyperparams.c_puct = 250
    hyperparams.lr = 0.1
    hyperparams.weight_decay = 1e-4
    hyperparams.training_tau = TrainingTau(fixed_tau_value=1)
    hyperparams.batch_size = 512

    # Set up an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    add_all_slots_to_arg_parser(parser, hyperparams)
    parser.add_argument('--ckpt_path', required=False)
    parser.add_argument('--debug', '-d', action='store_true', help='Open PDB at the end')
    parser.add_argument('--agent_eval_mode', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--show_game', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--show_action_counter', action=argparse.BooleanOptionalAction, default=False)

    # Parse the args and set the hyperparams
    args = parser.parse_args()

    load_from_arg_parser(args, hyperparams)

    G2048Multi = get_G2048Multi_game_class(hyperparams.device)

    training_run = AlphaMultiTrainingRunLightning.load_from_checkpoint(args.ckpt_path, map_location=args.device,
                                                                       game_class=G2048Multi, hyperparams=hyperparams)
    training_run.alpha_evaluator.device = args.device

    if args.show_game:
        training_run.game.listeners.append(CommandLineGame(0, extra_prints=[network_mcts_monitor]))
        training_run.agent.listeners.append(network_mcts_monitor)

    if args.show_action_counter:
        training_run.game.listeners.append(ActionCounterProgressBar(500))

    if args.agent_eval_mode:
        training_run.agent.eval()

    training_run.network.eval()
    load_from_arg_parser(args, training_run.agent.hyperparams)

    training_run.game.play()

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
