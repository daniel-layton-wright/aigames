import torch


def strip_nans(x):
    nans = torch.isnan(x).any(-1)
    while nans.dim() > 1:
        nans = nans.any(-1)

    return x[~nans]