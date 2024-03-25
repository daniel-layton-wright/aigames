"""
Opens a checkpoint, removes the dataset, and saves to specified location
"""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description='Remove dataset from checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--output', type=str, help='Path to the output checkpoint')
    parser.add_argument('--map_location', type=str, help='Map location for loading the model', default=None)
    args = parser.parse_args()

    checkpoint: dict = torch.load(args.checkpoint, map_location=args.map_location)
    checkpoint.pop('dataset')
    torch.save(checkpoint, args.output)


if __name__ == '__main__':
    main()
