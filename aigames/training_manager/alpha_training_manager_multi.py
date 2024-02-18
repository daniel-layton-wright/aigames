from ..agent.alpha_agent import BaseAlphaEvaluator
import torch
from ..agent.alpha_agent_multi import AlphaAgentMultiListener


class AlphaDatasetMulti(AlphaAgentMultiListener):
    def sample_minibatch(self, batch_size):
        raise NotImplementedError()


class BasicAlphaDatasetMulti(AlphaDatasetMulti):
    def __init__(self, evaluator: BaseAlphaEvaluator = None, max_size=50000, process_state=True, min_size=100):
        self.evaluator = evaluator
        self.max_size = max_size
        self.min_size = min_size
        self.states = None
        self.pis = None
        self.rewards = None
        self.process_state = process_state

        if process_state and (evaluator is None):
            raise ValueError('If process_state==True, you must give an evaluator.')

    def on_data_point(self, states: torch.Tensor, pis: torch.Tensor, rewards: torch.Tensor):
        if self.process_state:
            states = self.evaluator.process_state(states)

        if self.states is None:
            self.states = states
            self.pis = pis
            self.rewards = rewards
        else:
            self.states = torch.cat((self.states, states))
            self.pis = torch.cat((self.pis, pis))
            self.rewards = torch.cat((self.rewards, rewards))

        while len(self) > self.max_size:
            self.pop()

    def __len__(self):
        return self.states.shape[0] if self.states is not None else 0

    def pop(self):
        self.states = self.states[1:]
        self.pis = self.pis[1:]
        self.rewards = self.rewards[1:]

    def sample_minibatch(self, batch_size):
        dataset = TensorDataset(self.states, self.pis, self.rewards)
        sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=batch_size)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return next(iter(dataloader))

    def clear(self):
        self.states = None
        self.pis = None
        self.rewards = None


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
