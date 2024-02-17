from typing import Tuple, Final
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


def get_legal_action_masks_core_v1(states, legal_move_mask):
    lookup_tensor: Final = torch.tensor([16 ** 3, 16 ** 2, 16, 1], dtype=torch.float32, device=states.device).unsqueeze(1)
    ind = states.matmul(lookup_tensor).flatten().to(int)
    mask = legal_move_mask[ind, :].reshape(states.shape[0], 4, 2).any(dim=1)

    ind = lookup_tensor.T.matmul(states).flatten().to(int)
    return torch.cat([mask, legal_move_mask[ind, :].reshape(states.shape[0], 4, 2).any(dim=1)], dim=1)


def is_terminal_core_v1(states: torch.Tensor, legal_move_mask: torch.Tensor) -> torch.Tensor:
    legal_action_masks = get_legal_action_masks_core_v1(states, legal_move_mask)
    return legal_action_masks.sum(dim=1) == 0


def get_next_states_full_v1(states: torch.Tensor, actions: torch.Tensor, move_left_map: torch.Tensor,
                         move_right_map: torch.Tensor, reward_map: torch.Tensor, legal_move_mask: torch.Tensor)\
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    new_states, rewards = get_next_states_core_v1(states, actions, move_left_map, move_right_map, reward_map)

    # Set is terminal
    is_terminal = is_terminal_core_v1(new_states, legal_move_mask)

    env_is_next = torch.ones((states.shape[0],), dtype=torch.bool, device=states.device) & ~is_terminal

    return new_states, rewards, env_is_next, is_terminal


def get_next_states_core_v2(states: torch.Tensor, actions: torch.Tensor, move_left_map: torch.Tensor,
                            move_right_map: torch.Tensor, reward_map: torch.Tensor, ind: torch.Tensor,
                            indT: torch.Tensor, move_left_ind_map, move_right_ind_map)\
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lookup_tensor: Final = torch.tensor([16 ** 3, 16 ** 2, 16, 1], dtype=torch.float32, device=states.device).unsqueeze(1)

    new_states = torch.zeros_like(states)
    rewards = torch.zeros((states.shape[0], 1), dtype=torch.float32, device=states.device)
    next_ind = torch.zeros_like(ind)
    next_indT = torch.zeros_like(indT)

    left_mask = (actions == 0)
    right_mask = (actions == 1)
    up_mask = (actions == 2)
    down_mask = (actions == 3)

    left_lookup_idx = ind[left_mask].flatten()
    new_states[left_mask, :, :] = (move_left_map[left_lookup_idx, :].reshape((left_mask.sum(), 4, 4)))
    rewards[left_mask, :] = reward_map[left_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)
    next_ind[left_mask] = move_left_ind_map[left_lookup_idx, :].reshape(-1, 4, 1)

    right_lookup_idx = ind[right_mask].flatten()
    new_states[right_mask, :, :] = (move_right_map[right_lookup_idx, :].reshape((right_mask.sum(), 4, 4)))
    rewards[right_mask, :] = reward_map[right_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)
    next_ind[right_mask] = move_right_ind_map[right_lookup_idx, :].reshape(-1, 4, 1)

    next_indT[left_mask | right_mask] = lookup_tensor.T.matmul(states[left_mask | right_mask]).to(int)

    up_lookup_idx = indT[up_mask].flatten()
    new_states[up_mask, :, :] = (move_left_map[up_lookup_idx, :].reshape((up_mask.sum(), 4, 4))).transpose(1, 2)
    rewards[up_mask, :] = reward_map[up_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)
    next_indT[up_mask] = move_left_ind_map[up_lookup_idx, :].reshape(-1, 1, 4)

    down_lookup_idx = indT[down_mask].flatten()
    new_states[down_mask, :, :] = (
        move_right_map[down_lookup_idx, :].reshape((down_mask.sum(), 4, 4))).transpose(1, 2)
    rewards[down_mask, :] = reward_map[down_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)
    next_indT[down_mask] = move_right_ind_map[down_lookup_idx, :].reshape(-1, 1, 4)

    next_ind[up_mask | down_mask] = states[up_mask | down_mask].matmul(lookup_tensor).to(int)

    return new_states, rewards, next_ind, next_indT


def get_ind_core(states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    lookup_tensor: Final = torch.tensor([16 ** 3, 16 ** 2, 16, 1], dtype=torch.float32, device=states.device).unsqueeze(1)
    ind = states.matmul(lookup_tensor).to(int)
    indT = lookup_tensor.T.matmul(states).to(int)
    return ind, indT


def get_ind(states: torch.Tensor):
    if hasattr(states, 'ind') and hasattr(states, 'indT'):
        return states.ind, states.indT
    else:
        return get_ind_core(states)


def get_legal_action_masks_core_v2(states, legal_move_mask, ind, indT):
    mask = legal_move_mask[ind.flatten(), :].reshape(states.shape[0], 4, 2).any(dim=1)
    return torch.cat([mask, legal_move_mask[indT.flatten(), :].reshape(states.shape[0], 4, 2).any(dim=1)], dim=1)


def is_terminal_core_v2(states: torch.Tensor, legal_move_mask: torch.Tensor, ind, indT) -> torch.Tensor:
    legal_action_masks = get_legal_action_masks_core_v2(states, legal_move_mask, ind, indT)
    return legal_action_masks.sum(dim=1) == 0


def get_next_states_full_v2(states: torch.Tensor, actions: torch.Tensor, move_left_map: torch.Tensor,
                            move_right_map: torch.Tensor, reward_map: torch.Tensor, legal_move_mask: torch.Tensor,
                            move_left_ind_map, move_right_ind_map)\
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ind, indT = get_ind(states)

    new_states, rewards, new_ind, new_indT = get_next_states_core_v2(states, actions, move_left_map, move_right_map,
                                                                     reward_map, ind, indT, move_left_ind_map,
                                                                     move_right_ind_map)

    new_states.ind = new_ind
    new_states.new_indT = new_indT

    # Set is terminal
    is_terminal = is_terminal_core_v2(new_states, legal_move_mask, new_ind, new_indT)

    env_is_next = torch.ones((states.shape[0],), dtype=torch.bool, device=states.device) & ~is_terminal

    return new_states, rewards, env_is_next, is_terminal


def get_next_states_core_v3(states: torch.Tensor, actions: torch.Tensor, move_left_map: torch.Tensor,
                            move_right_map: torch.Tensor, reward_map: torch.Tensor, ind: torch.Tensor,
                            indT: torch.Tensor)\
        -> Tuple[torch.Tensor, torch.Tensor]:
    lookup_tensor: Final = torch.tensor([16 ** 3, 16 ** 2, 16, 1],
                                        dtype=torch.float32, device=states.device).unsqueeze(1)

    new_states = torch.zeros_like(states)
    rewards = torch.zeros((states.shape[0], 1), dtype=torch.float32, device=states.device)

    left_mask = (actions == 0)
    right_mask = (actions == 1)
    up_mask = (actions == 2)
    down_mask = (actions == 3)

    left_lookup_idx = ind[left_mask].flatten()
    new_states[left_mask, :, :] = (move_left_map[left_lookup_idx, :].reshape((left_mask.sum(), 4, 4)))
    rewards[left_mask, :] = reward_map[left_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    right_lookup_idx = ind[right_mask].flatten()
    new_states[right_mask, :, :] = (move_right_map[right_lookup_idx, :].reshape((right_mask.sum(), 4, 4)))
    rewards[right_mask, :] = reward_map[right_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    up_lookup_idx = indT[up_mask].flatten()
    new_states[up_mask, :, :] = (move_left_map[up_lookup_idx, :].reshape((up_mask.sum(), 4, 4))).transpose(1, 2)
    rewards[up_mask, :] = reward_map[up_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    down_lookup_idx = indT[down_mask].flatten()
    new_states[down_mask, :, :] = (
        move_right_map[down_lookup_idx, :].reshape((down_mask.sum(), 4, 4))).transpose(1, 2)
    rewards[down_mask, :] = reward_map[down_lookup_idx, :].reshape(-1, 4).sum(dim=1, keepdim=True)

    return new_states, rewards


def get_next_states_full_v3(states: torch.Tensor, actions: torch.Tensor, move_left_map: torch.Tensor,
                            move_right_map: torch.Tensor, reward_map: torch.Tensor, legal_move_mask: torch.Tensor,)\
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ind, indT = get_ind(states)

    new_states, rewards = get_next_states_core_v3(states, actions, move_left_map, move_right_map,
                                                  reward_map, ind, indT)

    new_ind, new_indT = get_ind(new_states)
    new_states.ind = new_ind
    new_states.new_indT = new_indT

    # Set is terminal
    is_terminal = is_terminal_core_v2(new_states, legal_move_mask, new_ind, new_indT)

    env_is_next = torch.ones((states.shape[0],), dtype=torch.bool, device=states.device) & ~is_terminal

    return new_states, rewards, env_is_next, is_terminal


def test_get_next_states_full(get_next_states_function, states, actions, move_left_map, move_right_map, reward_map, legal_move_mask,
                              **kwargs):
    for i in range(actions.shape[0]):
        cur_actions = actions[i]

        states, _, _, _ = get_next_states_function(states, cur_actions, move_left_map,
                                                   move_right_map, reward_map, legal_move_mask, **kwargs)


def main():
    import perftester as pt

    lookup_tensor = torch.tensor([16 ** 3, 16 ** 2, 16, 1], dtype=torch.float32).unsqueeze(1)
    move_left_ind_map = G2048Multi.MOVE_LEFT_MAP.matmul(lookup_tensor).long()
    move_right_ind_map = G2048Multi.MOVE_RIGHT_MAP.matmul(lookup_tensor).long()

    v1_results = pt.time_benchmark(test_get_next_states_full,
                                   get_next_states_function=get_next_states_full_v1,
                                   states=torch.randint(0, 10, (1000, 4, 4), dtype=torch.float32),
                                   actions=torch.randint(0, 4, (1000, 1000), dtype=torch.long),
                                   move_left_map=G2048Multi.MOVE_LEFT_MAP,
                                   move_right_map=G2048Multi.MOVE_RIGHT_MAP,
                                   reward_map=G2048Multi.REWARD_MAP,
                                   legal_move_mask=G2048Multi.LEGAL_MOVE_MASK,
                                   Number=1)

    v2_results = pt.time_benchmark(test_get_next_states_full,
                                   get_next_states_function=get_next_states_full_v2,
                                   states=torch.randint(0, 10, (1000, 4, 4), dtype=torch.float32),
                                   actions=torch.randint(0, 4, (1000, 1000), dtype=torch.long),
                                   move_left_map=G2048Multi.MOVE_LEFT_MAP,
                                   move_right_map=G2048Multi.MOVE_RIGHT_MAP,
                                   reward_map=G2048Multi.REWARD_MAP,
                                   legal_move_mask=G2048Multi.LEGAL_MOVE_MASK,
                                   move_left_ind_map=move_left_ind_map,
                                   move_right_ind_map=move_right_ind_map,
                                   Number=1)

    v3_results = pt.time_benchmark(test_get_next_states_full,
                                   get_next_states_function=get_next_states_full_v3,
                                   states=torch.randint(0, 10, (1000, 4, 4), dtype=torch.float32),
                                   actions=torch.randint(0, 4, (1000, 1000), dtype=torch.long),
                                   move_left_map=G2048Multi.MOVE_LEFT_MAP,
                                   move_right_map=G2048Multi.MOVE_RIGHT_MAP,
                                   reward_map=G2048Multi.REWARD_MAP,
                                   legal_move_mask=G2048Multi.LEGAL_MOVE_MASK,
                                   Number=1)

    print('v1:')
    print(pt.pp(v1_results))
    print('v2:')
    print(pt.pp(v2_results))
    print('v3:')
    print(pt.pp(v3_results))


if __name__ == '__main__':
    main()
