import argparse
import sys
import time
import torch
from .... import CommandLineGame
from ....agent.alpha_agent import TrainingTau
from ....agent.alpha_agent_multi import AlphaAgentMultiListener
from ....training_manager.alpha_training_manager_multi_lightning import AlphaMultiTrainingRunLightning
from aigames.training_manager.hyperparameters import AlphaMultiTrainingHyperparameters
from ....utils.listeners import ActionCounterProgressBar, MaxActionGameKiller
from .network_architectures import G2048MultiNetwork
from ....utils.utils import get_all_slots, add_all_slots_to_arg_parser, load_from_arg_parser
from ....game.G2048_multi import get_G2048Multi_game_class
from .G2048Multi_lightning import TrainingTauDecreaseOnPlateau, G2048TrainingRun
from .network_architectures import G2048MultiNetworkV2


class NetworkMCTSMonitor(AlphaAgentMultiListener):
    def __init__(self, pause_time=1):
        super().__init__()
        self.pi = None
        self.chosen_actions = None
        self.network_pi = None
        self.pause_time = pause_time
        self.network_value = None

    def after_mcts_search(self, mcts, pi, chosen_actions):
        self.pi = pi
        self.chosen_actions = chosen_actions
        self.network_pi = mcts.pi[:, 1].cpu().numpy()
        self.network_value = mcts.values[:, 1].cpu().numpy()

        print(str(self))
        time.sleep(self.pause_time)

    def __str__(self):
        # Print out the pi, the mcts search results, the chsen actions
        out = ''
        out += f'network pi: {self.network_pi}\n'
        out += f'network value: {self.network_value}\n'
        out += f'mcts results: {self.pi}\n'
        out += f'chosen_actions: {self.chosen_actions}\n'
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None, required=True, help='Path to a checkpoint to restore from')
    sysargv = sys.argv[1:]
    if '--help' in sysargv:
        sysargv.remove('--help')
    ckpt_path_args, _ = parser.parse_known_args(sysargv)

    training_run = G2048TrainingRun.load_from_checkpoint(ckpt_path_args.ckpt_path, map_location='cpu')
    hyperparams = training_run.hyperparams

    # Set up an arg parser which will look for all the slots in hyperparams
    parser = argparse.ArgumentParser()
    add_all_slots_to_arg_parser(parser, hyperparams)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--debug', '-d', action='store_true', help='Open PDB at the end')
    parser.add_argument('--agent_eval_mode', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--show_game', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--show_action_counter', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--max_actions', type=int, default=None)
    parser.add_argument('--pause_time', type=float, default=0.0)
    parser.add_argument('--network_class', type=str, default='G2048MultiNetworkV2')

    # Parse the args and set the hyperparams
    args = parser.parse_args()

    load_from_arg_parser(args, hyperparams)

    training_run = G2048TrainingRun.load_from_checkpoint(ckpt_path_args.ckpt_path, map_location='cpu',
                                                         hyperparams=hyperparams)

    if args.show_game:
        network_mcts_monitor = NetworkMCTSMonitor()
        training_run.game.listeners.append(CommandLineGame(args.pause_time))
        training_run.agent.listeners.append(network_mcts_monitor)

    if args.show_action_counter:
        training_run.game.listeners.append(ActionCounterProgressBar(500))

    if args.max_actions:
        training_run.game.listeners.append(MaxActionGameKiller(args.max_actions))

    if args.agent_eval_mode:
        training_run.agent.eval()

    training_run.network.eval()

    state = torch.FloatTensor([
        [1, 2, 4, 3],
        [4, 3, 6, 4],
        [1, 8, 5, 2],
        [2, 7, 1, 2]
    ])

    state = torch.FloatTensor([
        [10, 9, 2, 2],
        [8,  7, 5, 4],
        [6,  5, 3, 2],
        [2,  2, 1, 1]
    ])

    if args.debug:
        import pdb
        pdb.set_trace()

    training_run.agent.before_game_start(training_run.game)

    training_run.agent.get_actions(state.unsqueeze(0), torch.BoolTensor([True]))


if __name__ == '__main__':
    main()
