import torch


def compute_td_targets(state_values: torch.Tensor, rewards: torch.Tensor, td_lambda: float, discount: float):
    td_values = torch.zeros_like(state_values)
    td_values[-1] = rewards[-1]

    for i in range(len(state_values) - 2, -1, -1):
        td_values[i] = (rewards[i]
                        + discount * (1 - td_lambda) * state_values[i + 1]
                        + discount * td_lambda * td_values[i + 1])

    return td_values


def compute_td_targets_truncated_slow(state_values: torch.Tensor, rewards: torch.Tensor, td_lambda: float, discount: float,
                                      truncate_length: int):
    g = torch.zeros_like(state_values)
    for i in range(1, truncate_length+1):
        for j in range(i):
            g[i] += (discount**j) * rewards[j]

        g[i] += discount**i * state_values[i]

    out = torch.zeros_like(state_values[0])
    for i in range(1, truncate_length):
        out += (1 - td_lambda) * (td_lambda ** (i - 1)) * g[i]

    out += (td_lambda ** (truncate_length - 1)) * g[truncate_length]

    return out


def compute_td_targets_truncated(state_values: torch.Tensor, rewards: torch.Tensor, td_lambda: float, discount: float,
                                 truncate_length: int):
    if truncate_length >= state_values.shape[0]:
        return compute_td_targets(state_values, rewards, td_lambda, discount)[0]

    td_values = torch.zeros_like(state_values[:(truncate_length+1)])
    td_values[-1] = state_values[truncate_length]

    for i in range(truncate_length - 1, -1, -1):
        td_values[i] = (rewards[i]
                        + discount * (1 - td_lambda) * state_values[i + 1]
                        + discount * td_lambda * td_values[i + 1])

    return td_values[0]
