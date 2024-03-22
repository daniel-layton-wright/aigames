import math
from collections import Counter
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from aigames.agent.alpha_agent import BaseAlphaEvaluator
from aigames.agent.alpha_agent_multi import AlphaAgentMultiListener, Trajectory, AlphaAgentHyperparametersMulti
from aigames.utils.td import compute_td_targets, compute_td_targets_truncated


@dataclass(kw_only=True, slots=True)
class AlphaDatasetMultiHyperparameters(AlphaAgentHyperparametersMulti):
    batch_size: int = 32
    max_data_size: int = 10000000
    min_data_size: int = 1
    data_buffer_full_size: int = 32
    td_truncate_length: int = 10  # For PrioritizedTrajectoryDataset that uses truncated td returns
    dataset_device: str = 'cpu'


class AlphaDatasetMulti(AlphaAgentMultiListener):
    def __iter__(self):
        raise NotImplementedError()

    def to(self, device):
        raise NotImplementedError()

    @property
    def num_datapoints(self):
        raise NotImplementedError()

    @property
    def total_datapoints_seen(self):
        raise NotImplementedError()


class TrajectoryDataset(AlphaDatasetMulti):
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

        def yield_data(self, random=True, device=None):
            # Yield a random subset of the data of size self.batch_size
            batch_size = min(self.batch_size, self.data[0].shape[0])
            if random:
                order = np.random.permutation(self.data[0].shape[0])
            else:
                order = np.arange(self.data[0].shape[0])

            out = tuple(self.data[i][order[:batch_size]] for i in range(len(self.data)))
            self.data = [self.data[i][order[batch_size:]] for i in range(len(self.data))]

            if device is not None:
                out = tuple(x.to(device) for x in out)

            return out

        def __len__(self):
            return self.data[0].shape[0] if self.data[0] is not None else 0

    def __init__(self, evaluator: BaseAlphaEvaluator, hyperparams: AlphaDatasetMultiHyperparameters):
        super().__init__()
        self.trajectories = []
        self.hyperparams = hyperparams
        self.evaluator = evaluator
        self._total_datapoints_seen = 0

    def on_trajectories(self, trajectories: List[Trajectory]):
        # Process states
        for traj in trajectories:
            traj.states = self.evaluator.process_state(traj.states)
            traj.to(self.hyperparams.dataset_device)

        # Add to dataset
        self.trajectories.extend(trajectories)
        self._total_datapoints_seen += sum(len(traj.states) for traj in trajectories)
        self.enforce_max_size()

    @property
    def total_datapoints_seen(self):
        return self._total_datapoints_seen

    def __len__(self):
        return math.ceil(self.num_datapoints / float(self.hyperparams.batch_size))

    @property
    def num_datapoints(self):
        return sum(len(traj.states) for traj in self.trajectories)

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
        td_targets = compute_td_targets(state_values, cur_traj.rewards.to(state_values.device),
                                        self.hyperparams.td_lambda.get_lambda(),
                                        self.hyperparams.discount)
        return cur_traj.states, cur_traj.pis, td_targets

    def clear(self):
        self.trajectories = []

    def pop(self):
        self.trajectories.pop(0)

    def enforce_max_size(self):
        while sum(len(traj.states) for traj in self.trajectories) > self.hyperparams.max_data_size:
            self.pop()

    def states(self, device=None):
        if device is None:
            return torch.cat([traj.states for traj in self.trajectories])
        else:
            return torch.cat([traj.states.to(device) for traj in self.trajectories])

    def rewards(self, device=None):
        """
        Returns a M x R x N tensor of the rewards, padding the rewards to the max length
        where M is the number of trajectories, R is the max episode length, N is the number of players
        """
        if device is None:
            return torch.nn.utils.rnn.pad_sequence([traj.rewards for traj in self.trajectories], batch_first=True)
        else:
            return torch.nn.utils.rnn.pad_sequence([traj.rewards.to(device) for traj in self.trajectories], batch_first=True)

    def to(self, device):
        for traj in self.trajectories:
            traj.to(device)


class PrioritizedTrajectoryDataset(TrajectoryDataset):
    num_items = 4

    def __init__(self, evaluator: BaseAlphaEvaluator, hyperparams):
        super().__init__(evaluator, hyperparams)

    def on_trajectories(self, trajectories: List[Trajectory]):
        for traj in trajectories:
            # Fill in missing search values with network value
            missing = torch.isnan(traj.search_values).any(dim=1)
            self.evaluator.eval()
            network_result = self.evaluator.evaluate(traj.states[missing])
            self.evaluator.train()
            traj.search_values[missing] = network_result[1]

            discounted_rewards = compute_td_targets(traj.rewards, traj.rewards, 1, self.hyperparams.discount)
            # priority is the norm of the difference between the discounted_rewards and the mcts value
            diffs = discounted_rewards - traj.search_values
            traj.priorities = torch.norm(diffs, dim=1, p=1).cpu().numpy()

        super().on_trajectories(trajectories)

    def __getitem__(self, item):
        if len(self) == 0:
            return

        traj_lengths = np.array([len(traj.states) for traj in self.trajectories])
        traj_lengths_cum = np.cumsum(traj_lengths)
        traj_index = np.searchsorted(traj_lengths_cum, item, side='right')

        index_within_traj = item - traj_lengths_cum[traj_index - 1] if traj_index > 0 else item
        return self.get_data(self.trajectories[traj_index], index_within_traj)

    def get_all_priorities(self):
        return np.concatenate([traj.priorities for traj in self.trajectories])

    def __iter__(self):
        data_buffer = self.DataBuffer(num_items=self.num_items, batch_size=self.hyperparams.batch_size,
                                      full_size=self.hyperparams.data_buffer_full_size)

        for _ in range(len(self)):
            p = self.get_all_priorities()
            p /= p.sum()
            random_indices = np.random.choice(len(p), self.hyperparams.batch_size, p=p, replace=True)
            importance_sampling_weights = torch.tensor(1. / (len(p) * p[random_indices] + 0.001), dtype=torch.float32)
            random_indices = self.get_trajectory_and_sub_index(random_indices)

            all_states, sizes = self.get_states(random_indices)
            self.evaluator.eval()
            network_result = self.evaluator.evaluate(all_states)
            self.evaluator.train()

            first_state_indices = np.cumsum(sizes) - sizes
            first_states = all_states[first_state_indices]

            state_values = network_result[1]
            state_values = torch.split(state_values, sizes)
            state_values = torch.nn.utils.rnn.pad_sequence(state_values, batch_first=False)
            rewards = self.get_rewards(random_indices)

            td_targets = compute_td_targets_truncated(state_values, rewards, self.hyperparams.td_lambda.get_lambda(),
                                                      self.hyperparams.discount, self.hyperparams.td_truncate_length)

            pis = self.get_pis(random_indices)

            data_buffer.add_to_buffer(first_states, pis, td_targets, importance_sampling_weights)

            while data_buffer.full():
                yield data_buffer.yield_data(random=False)

        while len(data_buffer) > 0:
            yield data_buffer.yield_data(random=False)

    def get_states(self, indices):
        sizes = []
        all_states = []

        for (traj_index, index_within_traj) in indices:
            cur_states = self.trajectories[traj_index
                         ].states[index_within_traj:(index_within_traj + self.hyperparams.td_truncate_length + 1)]
            all_states.append(cur_states)
            sizes.append(len(cur_states))

        return torch.cat(all_states), sizes

    def get_rewards(self, indices):
        all_rewards = []

        for (traj_index, index_within_traj) in indices:
            cur_rewards = self.trajectories[traj_index
                          ].rewards[index_within_traj:(index_within_traj + self.hyperparams.td_truncate_length)]
            all_rewards.append(cur_rewards)

        return torch.nn.utils.rnn.pad_sequence(all_rewards, batch_first=False)

    def get_pis(self, indices):
        all_pis = []

        for (traj_index, index_within_traj) in indices:
            cur_pis = self.trajectories[traj_index].pis[[index_within_traj]]
            all_pis.append(cur_pis)

        return torch.cat(all_pis)

    def get_trajectory_and_sub_index(self, indices):
        traj_lengths = np.array([len(traj.states) for traj in self.trajectories])
        traj_lengths_cum = np.cumsum(traj_lengths)

        out = []

        for i in indices:
            traj_index = np.searchsorted(traj_lengths_cum, i, side='right')
            index_within_traj = i - traj_lengths_cum[traj_index - 1] if traj_index > 0 else i
            out.append((traj_index, index_within_traj))

        return out

    def sample_trajectory_index(self, n: int = 1):
        """
        Samples a trajectory index proportional to the sum of the priorities in each trajectory

        :param n: the number of trajectory indices to sample
        """
        priorities = self.get_trajectory_priorities()
        priorities /= priorities.sum()
        return np.random.choice(len(self.trajectories), n, p=priorities, replace=True)

    def sample_data(self, trajectory_indices):
        """

        """
        # Create a counter for each trajectory index to not repeat work
        trajectory_count = Counter(trajectory_indices)

        data_buffer = TrajectoryDataset.DataBuffer(num_items=self.num_items, batch_size=self.hyperparams.batch_size)

        for traj_ind, count in trajectory_count.items():
            for cur_data in self.sample_from_trajectory(traj_ind, n=count):
                data_buffer.add_to_buffer(*cur_data)

        return data_buffer.yield_data()

    def sample_from_trajectory(self, trajectory_index, n=1):
        """
        Samples data from the trajectory according to each data point's priority

        :param trajectory_index: the index of the trajectory to sample from
        :param n: the number of data points to sample
        """
        cur_traj = self.trajectories[trajectory_index]
        sampled_indices = np.random.choice(cur_traj.states.shape[0], n,
                                           p=(cur_traj.priorities / cur_traj.priorities.sum()), replace=True)

        for ind in sampled_indices:
            yield self.get_data(cur_traj, ind)

    def get_data(self, cur_traj, i):
        j = (i+self.hyperparams.td_truncate_length+1)
        self.evaluator.eval()
        network_result = self.evaluator.evaluate(cur_traj.states[i:j])
        self.evaluator.train()
        state_values = network_result[1]
        td_targets = compute_td_targets_truncated(state_values, cur_traj.rewards[i:j],
                                                  self.hyperparams.td_lambda.get_lambda(),
                                                  self.hyperparams.discount, self.hyperparams.td_truncate_length)
        return cur_traj.states[i], cur_traj.pis[i], td_targets

    def get_trajectory_priorities(self):
        """
        :return: an array of the priorities of the trajectories
        """
        return np.array([sum(traj.priorities) for traj in self.trajectories])


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
