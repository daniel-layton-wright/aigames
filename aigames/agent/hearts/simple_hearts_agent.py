import torch

from aigames import AgentMulti
from aigames.game.hearts import Hearts, HeartsHelper


class SimpleHeartsAgent(AgentMulti):
    """
    This agent implements a simple heuristic for playing hearts and works as follows:

    If the agent has cards of the suit led,
    * It will play the highest card it has under the highest card played so far
    * If no such card exists, it will play the lowest card it has

    If the agent does not have cards of the suit led, it will play the first of the following:
    * The queen of spades
    * The highest hearts it has
    * The highest card it has

    If the agent is leading the trick, it will play the lowest card it has

    """

    def get_actions(self, states: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        actions = torch.zeros(states.shape[0], dtype=torch.int64, device=states.device)
        legal_actions = Hearts.get_legal_action_masks(states)

        i = torch.arange(states.shape[0], device=states.device)

        # Get the games where current player is leading
        leading = (states[:, 0, 1] == 0)
        cur_player_index = states[:, 0, 0]
        cur_player_hand = (states[i, 0, 3:] == cur_player_index.unsqueeze(-1))

        # Lowest cards
        # Generate an index mask which as 13-1 repeated 4 times for the value of the card
        card_values = torch.arange(1, 14, device=states.device).repeat(4)
        card_values_desc = torch.arange(13, 0, -1, device=states.device).repeat(4)

        card_values_for_points = card_values.clone()
        card_values_for_points[HeartsHelper.hearts_mask.flatten().to(states.device)] += 13
        card_values_for_points[HeartsHelper.Card.QUEEN_OF_SPADES.value] = 27

        actions[leading] = (cur_player_hand[leading] * legal_actions[leading] * card_values_desc).argmax(dim=1)

        # Get the highest card in the suit led in the trick so far
        cur_trick_cards = (states[:, 4, 3:] > 0)
        card_led = (states[:, 4, 3:] == 1).int().argmax(dim=1)
        suit_led = card_led // 13
        suit_masks = HeartsHelper.suit_masks.to(states.device)[suit_led].squeeze(1)

        # If the player does not have the suit led, play the card with the highest points or highest card
        has_suit = (cur_player_hand * suit_masks).any(dim=1)
        actions[~leading & ~has_suit] = (cur_player_hand[~leading & ~has_suit]
                                        * legal_actions[~leading & ~has_suit] * card_values_for_points).argmax(dim=1)

        high_card_in_trick = (cur_trick_cards[~leading & has_suit] * suit_masks[~leading & has_suit] * card_values).argmax(dim=1).unsqueeze(-1) % 13
        actions[~leading & has_suit] = (
            ((high_card_in_trick > (card_values - 1)) * (card_values + 13)
                + (high_card_in_trick < (card_values - 1)) * card_values_desc)
            * legal_actions[~leading & has_suit]).argmax(dim=1)

        return actions
