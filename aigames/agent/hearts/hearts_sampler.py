import torch
from aigames.agent.alpha_agent_hidden import AlphaAgentHidden


class HeartsSampler(AlphaAgentHidden.Sampler):
    def __init__(self):
        pass
    
    def sample(self, hidden_state: torch.Tensor, pov_player_indices: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of Hearts hidden states, sample a corresponding batch of full states.

        :param hidden_state: Tensor of shape (batch_size, 5, 55) representing hidden game states.
        :return: Tensor of shape (batch_size, 5, 55) representing sampled full game states.
        """
        # Clone hidden states to create full states
        full_states = hidden_state.clone()

        # Count the number of cards each player has played
        # hidden_state[:, 2, 3:] contains the players who played each card
        # Convert hidden_state indices to long type to ensure compatibility with one_hot
        cards_per_player = torch.sum(
            torch.nn.functional.one_hot(hidden_state[:, 2, 3:].long(), num_classes=5)[:, :, 1:], dim=1
        )  # Shape: (batch_size, 4)

        # Calculate the number of cards remaining in each player's hand
        cards_in_hand = 13 - cards_per_player  # Shape: (batch_size, 4)

        # Exclude the POV player by setting their remaining cards to 0
        cards_in_hand.scatter_(1, pov_player_indices.unsqueeze(1), 0)

        # Generate player indices based on the number of cards in each player's hand
        player_range = torch.arange(1, 5, device=hidden_state.device).view(1, 4)
        player_indices = (player_range * (cards_in_hand > 0).long()).flatten()

        # Repeat player numbers according to the number of cards they hold
        player_indices = torch.repeat_interleave(player_indices, cards_in_hand.flatten())

        # Shuffle the player indices to randomly assign players to unknown cards
        shuffled_indices = player_indices[torch.randperm(player_indices.size(0))]

        # Create a mask for positions in full_states that need to be filled (-1)
        mask = full_states[:, 0, 3:] == -1  # Shape: (batch_size, 52)

        # Ensure the number of -1s matches the number of shuffled indices
        assert mask.sum() == shuffled_indices.size(0), "Mismatch between -1s and shuffled indices"

        # Assign the shuffled player indices to the masked positions
        full_states[:, 0, 3:].masked_scatter_(mask, shuffled_indices.to(full_states.dtype))

        return full_states
    
    def sample_multiple(self, hidden_states: torch.Tensor, player_indices: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Sample multiple full states for each hidden state in a batch.

        This method generates multiple samples of full game states for each hidden state
        in the input batch. It's useful for creating a diverse set of possible game states
        consistent with the partial information available in the hidden states.

        Args:
            hidden_states (torch.Tensor): A batch of hidden states from Hearts games.
                Shape: (batch_size, *state_shape)
            n_samples (int): The number of samples to generate for each hidden state.

        Returns:
            torch.Tensor: A tensor containing sampled full states.
                Shape: (batch_size, n_samples, *state_shape)

        Note:
            The returned tensor will have n_samples times as many states as the input,
            with each group of n_samples consecutive states corresponding to one input state.
        """        
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.repeat(
            n_samples, *[1] * (len(hidden_states.shape) - 1)
        )
        
        player_indices = player_indices.repeat(n_samples)
        
        out = self.sample(hidden_states, player_indices)
        return out.reshape(batch_size, n_samples, *out.shape[1:])
