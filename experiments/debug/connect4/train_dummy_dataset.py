from aigames.game.connect4 import *
from experiments.mp_alpha_connect4 import Connect4Network, Connect4Evaluator
from aigames.training_manager.alpha_training_manager import *
import pickle
import wandb
from aigames import Flatten


class Connect4Network2(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            Flatten()
        )

        self.base_out_features = 2688

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=7),
            nn.Softmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=1),
            nn.Tanh()
        )

    def forward(self, processed_state):
        base = self.base(processed_state)
        policy = self.policy_head(base).squeeze()
        value = self.value_head(base)
        return policy, value


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='data.pkl')
    parser.add_argument('--n_iters', type=int, default=1000)
    parser.add_argument('--minibatch_size', type=int, default=32)
    args = parser.parse_args()

    wandb.init(project='aigames2', tags=['debug', 'alpha'], tensorboard=True)

    with open(args.file, 'rb') as f:
        dataset: BasicAlphaDataset = pickle.load(f)

    dataset.max_size = 2
    while len(dataset) > dataset.max_size:
        dataset.pop()

    network = Connect4Network2()
    evaluator = Connect4Evaluator(network)
    optimizer = AlphaNetworkOptimizer(evaluator, torch.optim.Adam)

    r = range(args.n_iters)
    # r = tqdm(r)
    for _ in r:
        processed_states, pis, rewards = dataset.sample_minibatch(args.minibatch_size)

        pred_distns, pred_values = evaluator.network(processed_states)

        # print(processed_states)
        # print(pis)
        # print(rewards)
        # print(pred_distns)
        # print(pred_values)

        loss = optimizer.take_training_step_processed(processed_states, pis, rewards)
        wandb.log({'loss': loss})

    import pdb, sys
    sys.stdout = sys.stdout.terminal
    pdb.set_trace()


if __name__ == '__main__':
    main()
