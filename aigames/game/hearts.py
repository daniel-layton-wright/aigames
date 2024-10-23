import enum
from typing import Tuple, Final
from aigames.game.game_multi import GameMulti
import torch


class HeartsHelper:
    class Card(enum.Enum):
        TWO_OF_CLUBS = 0
        THREE_OF_CLUBS = 1
        FOUR_OF_CLUBS = 2
        FIVE_OF_CLUBS = 3
        SIX_OF_CLUBS = 4
        SEVEN_OF_CLUBS = 5
        EIGHT_OF_CLUBS = 6
        NINE_OF_CLUBS = 7
        TEN_OF_CLUBS = 8
        JACK_OF_CLUBS = 9
        QUEEN_OF_CLUBS = 10
        KING_OF_CLUBS = 11
        ACE_OF_CLUBS = 12
        TWO_OF_DIAMONDS = 13
        THREE_OF_DIAMONDS = 14
        FOUR_OF_DIAMONDS = 15
        FIVE_OF_DIAMONDS = 16
        SIX_OF_DIAMONDS = 17
        SEVEN_OF_DIAMONDS = 18
        EIGHT_OF_DIAMONDS = 19
        NINE_OF_DIAMONDS = 20
        TEN_OF_DIAMONDS = 21
        JACK_OF_DIAMONDS = 22
        QUEEN_OF_DIAMONDS = 23
        KING_OF_DIAMONDS = 24
        ACE_OF_DIAMONDS = 25
        TWO_OF_SPADES = 26
        THREE_OF_SPADES = 27
        FOUR_OF_SPADES = 28
        FIVE_OF_SPADES = 29
        SIX_OF_SPADES = 30
        SEVEN_OF_SPADES = 31
        EIGHT_OF_SPADES = 32
        NINE_OF_SPADES = 33
        TEN_OF_SPADES = 34
        JACK_OF_SPADES = 35
        QUEEN_OF_SPADES = 36
        KING_OF_SPADES = 37
        ACE_OF_SPADES = 38
        TWO_OF_HEARTS = 39
        THREE_OF_HEARTS = 40
        FOUR_OF_HEARTS = 41
        FIVE_OF_HEARTS = 42
        SIX_OF_HEARTS = 43
        SEVEN_OF_HEARTS = 44
        EIGHT_OF_HEARTS = 45
        NINE_OF_HEARTS = 46
        TEN_OF_HEARTS = 47
        JACK_OF_HEARTS = 48
        QUEEN_OF_HEARTS = 49
        KING_OF_HEARTS = 50
        ACE_OF_HEARTS = 51

        def __str__(self):
            """
            Should print the card as 2♣, etc.
            """
            numeric_value = self.value % 13 + 2

            # For numeric value > 10, we need to convert to J, Q, K, A
            if numeric_value > 10:
                numeric_value = "JQKA"[numeric_value - 11]

            suit = self.value // 13
            suit_str = "♣♦♠♥"[suit]
            return f"{numeric_value}{suit_str}"

    CLUBS = [Card.TWO_OF_CLUBS, Card.THREE_OF_CLUBS, Card.FOUR_OF_CLUBS, Card.FIVE_OF_CLUBS, Card.SIX_OF_CLUBS,
             Card.SEVEN_OF_CLUBS, Card.EIGHT_OF_CLUBS, Card.NINE_OF_CLUBS, Card.TEN_OF_CLUBS, Card.JACK_OF_CLUBS,
             Card.QUEEN_OF_CLUBS, Card.KING_OF_CLUBS, Card.ACE_OF_CLUBS]
    DIAMONDS = [Card.TWO_OF_DIAMONDS, Card.THREE_OF_DIAMONDS, Card.FOUR_OF_DIAMONDS, Card.FIVE_OF_DIAMONDS,
                Card.SIX_OF_DIAMONDS, Card.SEVEN_OF_DIAMONDS, Card.EIGHT_OF_DIAMONDS, Card.NINE_OF_DIAMONDS,
                Card.TEN_OF_DIAMONDS, Card.JACK_OF_DIAMONDS, Card.QUEEN_OF_DIAMONDS, Card.KING_OF_DIAMONDS,
                Card.ACE_OF_DIAMONDS]
    SPADES = [Card.TWO_OF_SPADES, Card.THREE_OF_SPADES, Card.FOUR_OF_SPADES, Card.FIVE_OF_SPADES, Card.SIX_OF_SPADES,
              Card.SEVEN_OF_SPADES, Card.EIGHT_OF_SPADES, Card.NINE_OF_SPADES, Card.TEN_OF_SPADES, Card.JACK_OF_SPADES,
              Card.QUEEN_OF_SPADES, Card.KING_OF_SPADES, Card.ACE_OF_SPADES]
    HEARTS = [Card.TWO_OF_HEARTS, Card.THREE_OF_HEARTS, Card.FOUR_OF_HEARTS, Card.FIVE_OF_HEARTS, Card.SIX_OF_HEARTS,
              Card.SEVEN_OF_HEARTS, Card.EIGHT_OF_HEARTS, Card.NINE_OF_HEARTS, Card.TEN_OF_HEARTS, Card.JACK_OF_HEARTS,
              Card.QUEEN_OF_HEARTS, Card.KING_OF_HEARTS, Card.ACE_OF_HEARTS]

    # noinspection PyTypeChecker
    clubs_mask: Final = torch.where((torch.arange(52) >= Card.TWO_OF_CLUBS.value)
                                    & (torch.arange(52) <= Card.ACE_OF_CLUBS.value),
                                    torch.ones(52), torch.zeros(52)).to(torch.bool).unsqueeze(0)
    # noinspection PyTypeChecker
    diamonds_mask: Final = torch.where((torch.arange(52) >= Card.TWO_OF_DIAMONDS.value)
                                       & (torch.arange(52) <= Card.ACE_OF_DIAMONDS.value),
                                       torch.ones(52), torch.zeros(52)).to(torch.bool).unsqueeze(0)
    # noinspection PyTypeChecker
    spades_mask: Final = torch.where((torch.arange(52) >= Card.TWO_OF_SPADES.value)
                                     & (torch.arange(52) <= Card.ACE_OF_SPADES.value),
                                     torch.ones(52), torch.zeros(52)).to(torch.bool).unsqueeze(0)
    # noinspection PyTypeChecker
    hearts_mask: Final = torch.where(torch.arange(52) >= Card.TWO_OF_HEARTS.value,
                                     torch.ones(52), torch.zeros(52)).to(torch.bool).unsqueeze(0)

    # you can index into the suit_masks by suit index to get the mask for that suit
    suit_masks: Final = torch.stack([clubs_mask, diamonds_mask, spades_mask, hearts_mask], dim=0)

    # noinspection PyTypeChecker
    two_of_clubs_mask: Final = torch.where(torch.arange(52) == Card.TWO_OF_CLUBS.value,
                                           torch.ones(52), torch.zeros(52)).to(torch.bool).unsqueeze(0)

    # noinspection PyTypeChecker
    rewards_mask: Final = (torch.where(hearts_mask, -1, 0)
                           + torch.where(torch.arange(52) == Card.QUEEN_OF_SPADES.value, -13, 0)).to(torch.float32)


def is_terminal_core(states):
    return states[:, 0, 2] == 52  # All 52 cards have been played


def get_next_states_core(states, actions, clubs_mask=HeartsHelper.clubs_mask, diamonds_mask=HeartsHelper.diamonds_mask,
                         spades_mask=HeartsHelper.spades_mask, hearts_mask=HeartsHelper.hearts_mask,
                         rewards_mask=HeartsHelper.rewards_mask):
    # Move all masks to the states device
    clubs_mask = clubs_mask.to(states.device)
    diamonds_mask = diamonds_mask.to(states.device)
    spades_mask = spades_mask.to(states.device)
    hearts_mask = hearts_mask.to(states.device)
    rewards_mask = rewards_mask.to(states.device)

    n = states.shape[0]
    r = torch.arange(n, device=states.device)
    next_states: torch.Tensor = states.clone()
    rewards = torch.zeros(n, 4, device=states.device)
    is_env = torch.zeros(n, dtype=torch.bool, device=states.device)

    # Update the current player
    cur_player = states[r, 0, 0]
    next_states[r, 0, 0] = (cur_player % 4) + 1

    # Update the number of cards played in the current trick
    cards_played_to_trick = states[:, 0, 1]
    last_card_in_trick = (cards_played_to_trick == 3)
    next_states[r, 0, 1] = (cards_played_to_trick + 1)

    # Update the number of cards played in the game
    cards_played_to_game = states[r, 0, 2]
    next_states[r, 0, 2] = cards_played_to_game + 1

    # Update the order in which cards were played
    next_states[r, 1, 3 + actions] = cards_played_to_game + 1
    next_states[r, 2, 3 + actions] = cur_player
    next_states[r, 4, 3 + actions] = cards_played_to_trick + 1

    # Remove the card from the player's hand
    next_states[r, 0, 3 + actions] = torch.tensor(0, dtype=torch.int8, device=states.device)

    # If this was the last card to the trick, we need to handle who won the card, the rewards, and update current
    # state to reflect new trick starting
    # We need to do this all in a vectorized fashion

    # First, we need to get which player won the trick for the tricks that are finished
    # The player who won the trick is the one who played the highest card of the suit led
    # Get the suit led
    cards_in_trick = next_states[last_card_in_trick, 4, 3:]
    led_card = (cards_in_trick == 1)
    clubs_led = (led_card * clubs_mask).any(dim=-1, keepdim=True)
    diamonds_led = (led_card * diamonds_mask).any(dim=-1, keepdim=True)
    spades_led = (led_card * spades_mask).any(dim=-1, keepdim=True)
    hearts_led = (led_card * hearts_mask).any(dim=-1, keepdim=True)

    cards_in_trick = cards_in_trick.to(torch.bool)

    in_suit_cards = (cards_in_trick * clubs_led * clubs_mask |
                     cards_in_trick * diamonds_led * diamonds_mask |
                     cards_in_trick * spades_led * spades_mask |
                     cards_in_trick * hearts_led * hearts_mask)

    winning_index = (torch.arange(52, device=states.device) * in_suit_cards).argmax(dim=-1) + 3
    winning_player = next_states[last_card_in_trick, 2, winning_index]

    # Update rewards of winning player
    rewards[last_card_in_trick, winning_player.to(torch.int) - 1] = (cards_in_trick * rewards_mask).sum(dim=-1)

    # All the cards in this trick go to the winning player
    next_states[last_card_in_trick, 3, 3:] += winning_player.unsqueeze(1) * cards_in_trick

    # The next player to play is the one who won the trick
    next_states[last_card_in_trick, 0, 0] = winning_player

    # Reset current trick
    next_states[last_card_in_trick, 0, 1] = 0
    next_states[last_card_in_trick, 4, 3:] = 0

    # Handle shooting the moon. If a player has all the hearts and the queen of spades, they're cumulative reward
    # for the game needs to be brought to 0 and everyone else gets -26
    points = (torch.stack([(next_states[:, 3, 3:] == (i + 1)).int() for i in range(4)]) * rewards_mask).sum(
        dim=-1).T
    # points is an N x 4 matrix where an entry in row i, column j is the points for player j in game i

    points *= (rewards != 0)  # for the purpose of rewards, only reward if they shoot the moon *this* trick

    # Get the index of the player who has all the hearts and the queen of spades
    moon_shooter = (points == -26).nonzero()
    cur_moon_shooter_rewards = rewards[moon_shooter[:, 0], moon_shooter[:, 1]]
    rewards[moon_shooter[:, 0], :] = -26
    rewards[moon_shooter[:, 0], moon_shooter[:, 1]] = 26 + cur_moon_shooter_rewards

    # TODO: maybe do early stopping if all the points have been taken

    is_terminal = is_terminal_core(next_states)

    return next_states, rewards, is_env, is_terminal


def get_legal_action_masks_core(states, clubs_mask=HeartsHelper.clubs_mask, diamonds_mask=HeartsHelper.diamonds_mask,
                                spades_mask=HeartsHelper.spades_mask, hearts_mask=HeartsHelper.hearts_mask,
                                two_of_clubs_mask=HeartsHelper.two_of_clubs_mask):
    # Move all masks to the states device
    clubs_mask = clubs_mask.to(states.device)
    diamonds_mask = diamonds_mask.to(states.device)
    spades_mask = spades_mask.to(states.device)
    hearts_mask = hearts_mask.to(states.device)
    two_of_clubs_mask = two_of_clubs_mask.to(states.device)

    # The base legal action mask is the cards in the current player's
    cur_player_hand = states[:, 0, 3:].eq(states[:, 0, 0].unsqueeze(1))

    first_move_of_game = (states[:, 0, 2] == 0)  # have to start with two of clubs

    cur_player_lead = (states[:, 0, 1] == 0)  # 0 cards played to current trick
    hearts_broken = ((states[:, 1, 3:] * hearts_mask) > 0).any(dim=-1)  # shape: n

    led_cards = (states[:, 4, 3:] == 1)  # shape: n x 52
    clubs_led = ((led_cards * clubs_mask) > 0).any(dim=-1)  # shape: n
    diamonds_led = ((led_cards * diamonds_mask) > 0).any(dim=-1)
    spades_led = ((led_cards * spades_mask) > 0).any(dim=-1)
    hearts_led = ((led_cards * hearts_mask) > 0).any(dim=-1)

    has_clubs = (cur_player_hand * clubs_mask).any(dim=-1)
    has_diamonds = (cur_player_hand * diamonds_mask).any(dim=-1)
    has_spades = (cur_player_hand * spades_mask).any(dim=-1)
    has_hearts = (cur_player_hand * hearts_mask).any(dim=-1)
    has_non_hearts = (has_clubs | has_diamonds | has_spades)

    #########################################################################
    # Do all calculations below here after figuring out necessary variables #
    #########################################################################
    legal_action_masks = cur_player_hand.clone()
    legal_action_masks[first_move_of_game] *= two_of_clubs_mask

    # Can't lead hearts until hearts has been broken or you only have hearts
    legal_action_masks[cur_player_lead & ~hearts_broken & has_non_hearts] *= (~hearts_mask)

    # Have to follow suit if possible
    legal_action_masks[clubs_led & has_clubs] *= clubs_mask
    legal_action_masks[diamonds_led & has_diamonds] *= diamonds_mask
    legal_action_masks[spades_led & has_spades] *= spades_mask
    legal_action_masks[hearts_led & has_hearts] *= hearts_mask

    return legal_action_masks


class Hearts(GameMulti):
    """
    The game of hearts. Actually this only implements a hand of hearts.
    The game is played with 4 players and 52 actions (one for each card)

    The state is a (5, 55) tensor

    The first two elements of the first row are:
        Whose turn it is (1-4)
        How many cards have already been played to the current trick
        How many cards have already been played in the game

    The row 0 is whose hand the card is in (1-4) or zero if not in anyone's hand

    The row 1 is the order in which cards were played (starting at 1) and zero if the card hasn't been played yet

    The row 2 is which player played the card

    The row 3 is who won the card

    The row 4 is the order in which cards were played in the current trick
    """
    device = 'cpu'

    get_next_states_jit = torch.jit.script(get_next_states_core)
    get_legal_action_masks_jit = torch.jit.script(get_legal_action_masks_core)  # trace wasn't working, not sure why
    is_terminal_jit = torch.jit.script(is_terminal_core)

    @classmethod
    def get_n_players(cls) -> int:
        return 4

    @classmethod
    def get_n_actions(cls) -> int:
        return 52

    @classmethod
    def get_state_shape(cls) -> Tuple[int, ...]:
        return 5, 55

    @classmethod
    def get_n_stochastic_actions(cls) -> int:
        return 0

    @classmethod
    def get_state_dtype(cls) -> torch.dtype:
        return torch.int8

    @classmethod
    def get_initial_states(cls, n) -> torch.Tensor:
        # Start by generating an n x 52 tensor that is a random permutation of 13 1s, 2s, 3s, 4s
        # This will represent the cards in the hands of the players
        hands = (1 + torch.arange(4, dtype=torch.int8, device=cls.device)).repeat(n, 13)
        random_perm = torch.argsort(torch.rand(*hands.shape), dim=-1)
        hands = hands[torch.arange(hands.shape[0]).unsqueeze(-1), random_perm]
        player_with_2clubs = hands[:, HeartsHelper.Card.TWO_OF_CLUBS.value]

        states = torch.zeros(n, 5, 55, dtype=torch.int8, device=cls.device)
        states[:, 0, 3:] = hands
        states[:, 0, 0] = player_with_2clubs  # two of clubs goes first
        return states

    @classmethod
    def get_cur_player_index(cls, states) -> torch.Tensor:
        return states[:, 0, 0] - 1

    @classmethod
    def get_next_states(cls, states: torch.Tensor, actions: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.BoolTensor, torch.BoolTensor]:
        """
        :return: next_states, rewards, is_env, is_terminal
        """
        return get_next_states_core(states, actions)

    @classmethod
    def is_terminal(cls, states):
        return is_terminal_core(states)

    @classmethod
    def get_legal_action_masks(cls, states):
        return get_legal_action_masks_core(states)

    def __str__(self):
        return '\n'.join([Hearts.state_to_str(state) for state in self.states])

    @classmethod
    def state_to_str(cls, state):
        out = ""
        import pandas as pd
        from termcolor import colored

        # Create a DataFrame to hold the player information
        columns = ['Turn', 'Name', 'Score', 'Current', 'Clubs', 'Diamonds', 'Spades', 'Hearts']
        data = []

        legal_action_masks = Hearts.get_legal_action_masks(state.unsqueeze(0)).squeeze(0)

        # Extract information for each player
        for pidx in range(4):
            player_name = f"Player {pidx + 1}"
            # Sum of rewards for this player
            current_score = (state[3, 3:].eq(pidx + 1) * HeartsHelper.rewards_mask.to(state.device)).sum().int().item()
            cards_in_hand = state[0, 3:] == (pidx + 1)  # Assuming cards in hand starts from index 3
            is_current_player = state[0, 0] == pidx + 1

            # Find the card played to the current trick by this player
            current_trick_card_idx = (state[2, 3:] * (state[4, 3:] > 0).int()).eq(pidx + 1).nonzero(as_tuple=True)[0]
            current_trick_card = str(HeartsHelper.Card(
                current_trick_card_idx.item())) if current_trick_card_idx.numel() > 0 else ""

            # Mark the current player
            turn_indicator = '*' if is_current_player else ''
            player_name_colored = player_name  # colored(player_name, 'green') if is_current_player else player_name

            # Split cards in hand by suits with legal actions colored green
            clubs = ' '.join([
                colored(str(HeartsHelper.Card(card_idx)), 'green') if legal_action_masks[card_idx]
                else str(HeartsHelper.Card(card_idx)) for card_idx, present in enumerate(cards_in_hand[:13]) if present
            ])
            diamonds = ' '.join([
                colored(str(HeartsHelper.Card(card_idx + 13)), 'green') if legal_action_masks[card_idx + 13]
                else str(HeartsHelper.Card(card_idx + 13))
                for card_idx, present in enumerate(cards_in_hand[13:26]) if present
            ])
            spades = ' '.join([
                colored(str(HeartsHelper.Card(card_idx + 26)), 'green')
                if legal_action_masks[card_idx + 26]
                else str(HeartsHelper.Card(card_idx + 26))
                for card_idx, present in enumerate(cards_in_hand[26:39]) if present
            ])
            hearts = ' '.join([
                colored(str(HeartsHelper.Card(card_idx + 39)), 'green')
                if legal_action_masks[card_idx + 39]
                else str(HeartsHelper.Card(card_idx + 39))
                for card_idx, present in enumerate(cards_in_hand[39:]) if present
            ])

            # Append the row for this player
            data.append([
                turn_indicator, player_name_colored, current_score, colored(current_trick_card, 'yellow'),
                clubs, diamonds, spades, hearts
            ])

        # Create the DataFrame
        df = pd.DataFrame(data, columns=columns)

        # Now we'll custom print the rows of the dataframe to align them
        # First get the max length of each column
        import re

        def terminal_len(s):
            # length of string when stripping out formatting characters
            return len(re.sub(
                r'[\u001B\u009B][\[\]()#;?]*((([a-zA-Z\d]*(;[-a-zA-Z\d\/#&.:=?%@~_]*)*)?\u0007)'
                r'|((\d{1,4}(?:;\d{0,4})*)?[\dA-PR-TZcf-ntqry=><~]))',
                '', s))

        def terminal_ljust(s, ljustlen):
            # ljust string accounting for formatting ansi characters
            offset = len(s) - terminal_len(s)
            return s.ljust(offset + ljustlen)

        max_len = df.applymap(lambda x: terminal_len(str(x))).max()

        # Iterate over the rows of the dataframe
        for row in df.itertuples():
            row = row._asdict()
            # Print the row with the correct padding
            out += '  '.join([terminal_ljust(str(row[col]), max_len[col]) for col in columns]) + '\n'

        # Add a line of --- under the state
        out += '-' * (2*len(columns) + sum(max_len[col] for col in columns)) + '\n'

        out += str(state) + '\n'

        return out

    @classmethod
    def action_to_str(cls, action):
        # convert tensor to int if necessary
        if isinstance(action, torch.Tensor):
            action = action.item()
        return str(HeartsHelper.Card(action))


class HeartsCuda(Hearts):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_hearts_game_class(device):
    return HeartsCuda if device == 'cuda' else Hearts
