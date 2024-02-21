import torch
from ..agent import AgentMulti


class SimpleG2048Agent(AgentMulti):
    def __init__(self):
        super().__init__()
        self.game = None
        self.last_move_was_up = None

    def before_game_start(self, game):
        self.game = game
        self.last_move_was_up = None

    def get_actions(self, states: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        if self.last_move_was_up is None:
            self.last_move_was_up = torch.zeros(mask.shape, dtype=torch.bool, device=mask.device)

        # Start with zeros: go left if possible
        out = torch.zeros((states.shape[0],), dtype=torch.float32, device=states.device)
        legal_action_masks = self.game.get_legal_action_masks(states)

        left_illegal = (legal_action_masks[:, 0] == 0)
        right_illegal = (legal_action_masks[:, 1] == 0)
        down_illegal = (legal_action_masks[:, 3] == 0)

        out[left_illegal & ~down_illegal] = 3  # If can't go left go down
        out[left_illegal & down_illegal & ~right_illegal] = 1  # If can't go left or down, go right
        out[left_illegal & down_illegal & right_illegal] = 2  # If can't go left, down, or right, go up

        down_after_up = self.last_move_was_up[mask] & ~down_illegal
        out[down_after_up] = 3

        self.last_move_was_up[mask][out == 2] = True

        return out
