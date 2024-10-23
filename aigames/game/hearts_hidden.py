import torch
from .hearts import Hearts


class HeartsHidden(Hearts):
    """
    Partially observable version of the Hearts game where players cannot see 
    the cards in other players' hands. Other players' card indicators are 
    replaced with -1 to conceal their holdings.
    """

    @classmethod
    def get_observable_states(cls, states: torch.Tensor, player_indices: torch.Tensor) -> torch.Tensor:
        """
        Generate observable states with hidden information for each player.

        This method masks the card ownership of all players except the current 
        player by replacing their indicators with -1.

        :param states: Tensor containing the current game states.
        :param player_indices: Tensor containing the indices of the current players.
        
        :return: Tensor of observable states with other players' cards hidden.
        """
        # Clone the states to avoid mutating the original tensor
        observable_states = states.clone()

        # The card ownership information starts from index 3
        # Replace other players' card ownership with -1.
        # The following operation sets ownership to -1 where the ownership 
        # does not match the current player.
        mask = ((observable_states[:, 0, 3:] != (player_indices.unsqueeze(1) + 1))
                & (observable_states[:, 0, 3:] != 0))
        observable_states[:, 0, 3:][mask] = -1

        return observable_states
