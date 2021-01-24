from ..game import GameListener
import torch.utils.data


class TrainingListener(GameListener):
    def before_begin_training(self, training_manager):
        pass

    def on_training_step(self, iter: int, loss: float, training_manager, **kwargs):
        pass


class ListDataset(torch.utils.data.Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(l) for l in lists)
        self.lists = lists

    def __getitem__(self, index):
        return tuple(lists[index] for lists in self.lists)

    def __len__(self):
        return len(self.lists[0])