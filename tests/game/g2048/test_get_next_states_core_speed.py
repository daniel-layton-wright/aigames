from typing import Final, Tuple
import torch
from aigames.game.G2048_multi import G2048Multi


def get_next_states_core_v1(states: torch.Tensor, actions: torch.Tensor, move_left_map: torch.Tensor,
                         move_right_map: torch.Tensor, reward_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    lookup_tensor: Final = torch.tensor([16 ** 3, 16 ** 2, 16, 1], dtype=torch.float32, device=states.device).unsqueeze(1)
    new_states = torch.zeros_like(states)
    rewards = torch.zeros((states.shape[0], 1), dtype=torch.float32, device=states.device)
    left_mask = (actions == 0)
    right_mask = (actions == 1)
    up_mask = (actions == 2)
    down_mask = (actions == 3)

    left_lookup_idx = states[left_mask, :, :].matmul(lookup_tensor).flatten().to(int)
    new_states[left_mask, :, :] = (move_left_map[left_lookup_idx, :].reshape((left_mask.sum(), 4, 4)))
    rewards[left_mask, :] = reward_map[left_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    right_lookup_idx = states[right_mask, :, :].matmul(lookup_tensor).flatten().to(int)
    new_states[right_mask, :, :] = (move_right_map[right_lookup_idx, :].reshape((right_mask.sum(), 4, 4)))
    rewards[right_mask, :] = reward_map[right_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    up_lookup_idx = lookup_tensor.T.matmul(states[up_mask, :, :]).flatten().to(int)
    new_states[up_mask, :, :] = (move_left_map[up_lookup_idx, :].reshape((up_mask.sum(), 4, 4))).transpose(1, 2)
    rewards[up_mask, :] = reward_map[up_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    down_lookup_idx = lookup_tensor.T.matmul(states[down_mask, :, :]).flatten().to(int)
    new_states[down_mask, :, :] = (
        move_right_map[down_lookup_idx, :].reshape((down_mask.sum(), 4, 4))).transpose(1, 2)
    rewards[down_mask, :] = reward_map[down_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    # This can be used to give a reward for keeping highest tile in bottom left. 512 is the reward value
    # rewards += ((new_states[:, 3, 0] == new_states.amax(dim=(1, 2))) * 512).unsqueeze(1)

    return new_states, rewards


def get_next_states_core_v2(states: torch.Tensor, actions: torch.Tensor, move_left_map: torch.Tensor,
                            move_right_map: torch.Tensor, reward_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    lookup_tensor: Final = torch.tensor([16 ** 3, 16 ** 2, 16, 1], dtype=torch.float32, device=states.device).unsqueeze(1)
    new_states = torch.zeros_like(states)
    rewards = torch.zeros((states.shape[0], 1), dtype=torch.float32, device=states.device)
    left_mask = (actions == 0)
    right_mask = (actions == 1)
    up_mask = (actions == 2)
    down_mask = (actions == 3)

    lookup_idx = states[left_mask | right_mask].matmul(lookup_tensor).to(int)
    lookup_idx_up_down = lookup_tensor.T.matmul(states[up_mask | down_mask]).to(int)

    left_lookup_idx = lookup_idx[actions[left_mask | right_mask] == 0].flatten()
    new_states[left_mask, :, :] = (move_left_map[left_lookup_idx, :].reshape((left_mask.sum(), 4, 4)))
    rewards[left_mask, :] = reward_map[left_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    right_lookup_idx = lookup_idx[actions[left_mask | right_mask] == 1].flatten()
    new_states[right_mask, :, :] = (move_right_map[right_lookup_idx, :].reshape((right_mask.sum(), 4, 4)))
    rewards[right_mask, :] = reward_map[right_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    up_lookup_idx = lookup_idx_up_down[actions[up_mask | down_mask] == 2].flatten()
    new_states[up_mask, :, :] = (move_left_map[up_lookup_idx, :].reshape((up_mask.sum(), 4, 4))).transpose(1, 2)
    rewards[up_mask, :] = reward_map[up_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    down_lookup_idx = lookup_idx_up_down[actions[up_mask | down_mask] == 3].flatten()
    new_states[down_mask, :, :] = (
        move_right_map[down_lookup_idx, :].reshape((down_mask.sum(), 4, 4))).transpose(1, 2)
    rewards[down_mask, :] = reward_map[down_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    return new_states, rewards


def main():
    import perftester as pt

    v1_results = pt.time_benchmark(get_next_states_core_v1,
                                   states=torch.randint(0, 10, (1000, 4, 4), dtype=torch.float32),
                                   actions=torch.randint(0, 4, (1000,), dtype=torch.long),
                                   move_left_map=G2048Multi.MOVE_LEFT_MAP,
                                   move_right_map=G2048Multi.MOVE_RIGHT_MAP,
                                   reward_map=G2048Multi.REWARD_MAP,
                                   Number=1000)

    v2_results = pt.time_benchmark(get_next_states_core_v2,
                                   states=torch.randint(0, 10, (1000, 4, 4), dtype=torch.float32),
                                   actions=torch.randint(0, 4, (1000,), dtype=torch.long),
                                   move_left_map=G2048Multi.MOVE_LEFT_MAP,
                                   move_right_map=G2048Multi.MOVE_RIGHT_MAP,
                                   reward_map=G2048Multi.REWARD_MAP,
                                   Number=1000)

    print('v1:')
    print(pt.pp(v1_results))
    print('v2:')
    print(pt.pp(v2_results))


if __name__ == '__main__':
    main()