"""
Util script for converting a pytorch network to a ONNX network
"""
import pytorch_lightning as pl
import torch
import argparse
from torch import nn
from aigames.game.game_multi import GameMulti
from aigames.utils.utils import import_string
import sys

global_parser = argparse.ArgumentParser()
global_parser.add_argument('--import_module', type=str, help='Module to import from in order to load checkpoint',
                    required=False, default=None)
args, _ = global_parser.parse_known_args(sys.argv[1:])
if args.import_module is not None:
    exec(f'from {args.import_module} import *')


def main():
    parser = argparse.ArgumentParser(description='Convert a pytorch network to a ONNX network')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--output', type=str, help='Path to the output ONNX network')
    parser.add_argument('--map_location', type=str, help='Map location for loading the model', default=None)
    parser.add_argument('--module_class', type=str, help='Class of the module')
    parser.add_argument('--game_class', type=str, help='Class of the game')
    parser.add_argument('--import_module', type=str, help='Module to import from in order to load checkpoint',
                        required=False, default=None)
    args = parser.parse_args()

    module_class: pl.LightningModule = import_string(args.module_class)
    module = module_class.load_from_checkpoint(args.checkpoint, map_location=args.map_location)
    network = module.network
    network.eval()
    network.to('cpu')

    game_class: GameMulti = import_string(args.game_class)
    example_state = game_class.get_initial_states(1)
    example_input = network.process_state(example_state)
    torch.onnx.export(network, example_input, args.output)


if __name__ == '__main__':
    main()
