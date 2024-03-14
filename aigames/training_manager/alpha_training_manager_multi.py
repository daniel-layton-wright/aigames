import math
from typing import List
import numpy as np
from ..agent.alpha_agent import BaseAlphaEvaluator
import torch
from ..agent.alpha_agent_multi import AlphaAgentMultiListener, Trajectory
import pytorch_lightning as pl


class BasicAlphaDatasetMulti(AlphaAgentMultiListener):
    def __init__(self, evaluator: BaseAlphaEvaluator = None, max_size=50000, process_state=True, min_size=100):
        self.evaluator = evaluator
        self.max_size = max_size
        self.min_size = min_size

        self.data = {
            'states': None,
            'pis': None,
            'rewards': None
        }

        self.process_state = process_state

        if process_state and (evaluator is None):
            raise ValueError('If process_state==True, you must give an evaluator.')

    def on_data_point(self, states: torch.Tensor, pis: torch.Tensor, rewards: torch.Tensor, *args, **kwargs):
        if self.process_state:
            states = self.evaluator.process_state(states)

        data_names_and_values = self.get_data_names_and_values(states, pis, rewards, *args, **kwargs)

        if self.data['states'] is None:
            for key, val in data_names_and_values:
                self.data[key] = val
        else:
            for key, val in data_names_and_values:
                self.data[key] = torch.cat((self.data[key], val))

        self.enforce_max_size()

    def get_data_names_and_values(self, states, pis, rewards, *args, **kwargs):
        return [('states', states), ('pis', pis), ('rewards', rewards)]

    def enforce_max_size(self):
        while len(self) > self.max_size:
            self.pop()

    def __len__(self):
        return self.data['states'].shape[0] if self.data['states'] is not None else 0

    def pop(self):
        for key in self.data:
            self.data[key] = self.data[key][1:]

    def sample_minibatch(self, batch_size):
        dataset = TensorDataset(*(self.data[key] for key in self.data))  # TODO : make sure this is in the right order
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=batch_size)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return next(iter(dataloader))

    def clear(self):
        for key in self.data:
            self.data[key] = None

    def __getattr__(self, item):
        if item in self.data:
            return self.data[item]
        else:
            return super().__getattr__(item)


class TensorDataset(torch.utils.data.Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensors[index] for tensors in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]


def compute_td_targets(state_values: torch.Tensor, rewards: torch.Tensor, td_lambda: float, discount: float):
    td_values = torch.zeros_like(state_values)
    td_values[-1] = rewards[-1]

    for i in range(len(state_values) - 2, -1, -1):
        td_values[i] = (rewards[i]
                        + discount * (1 - td_lambda) * state_values[i + 1]
                        + discount * td_lambda * td_values[i + 1])

    return td_values


class TrajectoryDataset(pl.LightningDataModule, AlphaAgentMultiListener):
    num_items = 3

    class DataBuffer:
        def __init__(self, num_items, batch_size, full_size=None):
            self.data = [None for _ in range(num_items)]
            self.batch_size = batch_size
            self.full_size = max(full_size, batch_size) if full_size is not None else batch_size

        def add_to_buffer(self, *args):
            for i, arg in enumerate(args):
                self.data[i] = arg if self.data[i] is None else torch.cat((self.data[i], arg))

        def full(self):
            return self.data[0].shape[0] >= self.full_size

        def yield_data(self):
            # Yield a random subset of the data of size self.batch_size
            batch_size = min(self.batch_size, self.data[0].shape[0])
            random_order = np.random.permutation(self.data[0].shape[0])

            out = tuple(self.data[i][random_order[:batch_size]] for i in range(len(self.data)))
            self.data = [self.data[i][random_order[batch_size:]] for i in range(len(self.data))]

            return out

        def __len__(self):
            return self.data[0].shape[0] if self.data[0] is not None else 0

    def __init__(self, evaluator: BaseAlphaEvaluator, hyperparams):
        super().__init__()
        self.trajectories = []
        self.hyperparams = hyperparams
        self.evaluator = evaluator
        self.total_datapoints_seen = 0

    def on_trajectories(self, trajectories: List[Trajectory]):
        # Process states
        for traj in trajectories:
            traj.states = self.evaluator.process_state(traj.states)

        # Add to dataset
        self.trajectories.extend(trajectories)
        self.total_datapoints_seen += sum(len(traj.states) for traj in trajectories)
        self.enforce_max_size()

    def __len__(self):
        return math.ceil(sum(len(traj.states) for traj in self.trajectories) / float(self.hyperparams.batch_size))

    def train_dataloader(self):
        return self

    def __iter__(self):
        random_order_of_trajectories = np.random.permutation(self.trajectories)
        data_buffer = self.DataBuffer(num_items=self.num_items, batch_size=self.hyperparams.batch_size,
                                      full_size=self.hyperparams.data_buffer_full_size)

        for cur_traj in random_order_of_trajectories:
            data_buffer.add_to_buffer(*self.get_data(cur_traj))

            while data_buffer.full():
                yield data_buffer.yield_data()

        while len(data_buffer) > 0:
            yield data_buffer.yield_data()

    def get_data(self, cur_traj):
        self.evaluator.eval()
        network_result = self.evaluator.evaluate(cur_traj.states)
        self.evaluator.train()
        state_values = network_result[1]
        td_targets = compute_td_targets(state_values, cur_traj.rewards, self.hyperparams.td_lambda.get_lambda(),
                                        self.hyperparams.discount)
        return cur_traj.states, cur_traj.pis, td_targets

    def clear(self):
        self.trajectories = []

    def pop(self):
        self.trajectories.pop(0)

    def enforce_max_size(self):
        while sum(len(traj.states) for traj in self.trajectories) > self.hyperparams.max_data_size:
            self.pop()

    @property
    def states(self):
        return torch.cat([traj.states for traj in self.trajectories])


class NumMovesTrajectoryDataset(TrajectoryDataset):
    num_items = 4

    def get_data(self, cur_traj):
        self.evaluator.eval()
        network_result = self.evaluator.evaluate(cur_traj.states)
        self.evaluator.train()
        state_values = network_result[1]
        num_moves = network_result[2]
        td_targets = compute_td_targets(state_values, cur_traj.rewards, self.hyperparams.td_lambda.get_lambda(),
                                        self.hyperparams.discount)

        num_moves_target = compute_td_targets(num_moves,
                                              torch.ones_like(num_moves),
                                              self.hyperparams.num_moves_td_lambda.get_lambda(),
                                              1.0)

        return cur_traj.states, cur_traj.pis, td_targets, num_moves_target
