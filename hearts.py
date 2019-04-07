from enum import Enum
from itertools import product
import torch
import numpy as np
import copy
from game import *


# Suit = Enum('Suit', 'CLUBS DIAMONDS HEARTS SPADES', qualname = __name__)


# Value = Enum('Value', 'TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN JACK QUEEN KING ACE', qualname = __name__)


class Suit(Enum):
    CLUBS = 1
    DIAMONDS = 2
    SPADES = 3
    HEARTS = 4

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Value(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __lt__(self, other):
        return self.value < other.value


class Card:
    def __init__(self, value : Value, suit : Suit):
        self.value = value
        self.suit = suit

    def __repr__(self):
        return self.value.name + ' of ' + self.suit.name

    def __eq__(self, other):
        return self.value == other.value and self.suit == other.suit

    def __hash__(self):
        return hash((self.value, self.suit))

    def __lt__(self, other):
        return self.suit < other.suit or (self.suit == other.suit and self.value < other.value)


STANDARD_DECK = [Card(value, suit) for suit, value in product(Suit, Value)]


INDEX_IN_STANDARD_DECK = {card: i for i, card in enumerate(STANDARD_DECK)}


ONE_HOT_TENSOR_BY_CARD = {card: torch.FloatTensor((np.arange(len(STANDARD_DECK)) == i).astype(int)) for i, card in enumerate(STANDARD_DECK)}


class HeartsState:
    def __init__(self, player_hands, current_trick_cards, current_trick_leader, current_player, past_trick_cards, player_scores, prev_trick_scores):
        self.player_hands = player_hands
        self.current_trick_cards = current_trick_cards  # list where current_trick_cards[i] is the i'th player's played card
        self.current_trick_leader = current_trick_leader
        self.current_player = current_player
        self.past_trick_cards = past_trick_cards
        self.player_scores = player_scores
        self.prev_trick_scores = prev_trick_scores

    @staticmethod
    def initial_state(n_players = 4):
        shuffled_cards = np.random.permutation(STANDARD_DECK)
        player_hands = [sorted(hand) for hand in shuffled_cards.reshape(n_players, -1).tolist()]
        current_trick_cards = [None] * n_players
        # player who has the two of clubs
        current_player = [i for i, hand in enumerate(player_hands) if Card(Value.TWO, Suit.CLUBS) in hand][0]
        past_trick_cards = []
        player_scores = np.zeros(n_players)

        return HeartsState(player_hands, current_trick_cards, current_player, current_player, past_trick_cards, player_scores, None)


class PartiallyObservedHeartsState:
    def __init__(self, own_hand, current_trick_cards, current_trick_leader, is_first_trick, is_blood_drawn):
        self.own_hand = own_hand
        self.current_trick_cards = current_trick_cards
        self.current_trick_leader = current_trick_leader
        self.is_first_trick = is_first_trick
        self.is_blood_drawn = is_blood_drawn


class Hearts(PartiallyObservableSequentialGame):
    def __init__(self, players):
        super().__init__(players)
        self.players = players
        self.state = HeartsState.initial_state()

    @staticmethod
    def legal_actions(state : PartiallyObservedHeartsState):
        if state.is_first_trick and all(x is None for x in state.current_trick_cards):
            # This is the first trick, must play 2 of clubs
            return [Card(Value.TWO, Suit.CLUBS)]
        elif all(x is None for x in state.current_trick_cards):
            # New trick starting
            if state.is_blood_drawn:
                return state.own_hand
            else:
                non_hearts_cards = [card for card in state.own_hand if card.suit != Suit.HEARTS]
                if len(non_hearts_cards) == 0:
                    # This player only has hearts, so they are allowed to lead hearts
                    return state.own_hand
                else:
                    # Must lead non-hearts if possible, since hearts have not been broken
                    return non_hearts_cards
        else:
            # Trick already started
            legal_cards = []
            led_suit = state.current_trick_cards[state.current_trick_leader].suit
            for card in state.own_hand:
                if card.suit == led_suit:
                    legal_cards.append(card)

            if len(legal_cards) > 0:
                return legal_cards
            else:
                # Player doesn't have the suit, all cards are fair play
                return state.own_hand

    @staticmethod
    def get_player_index(state : HeartsState):
        return state.current_player

    @staticmethod
    def get_next_state(state : HeartsState, action : Card):
        new_state = copy.deepcopy(state)

        # remove the card from the player's hand
        new_state.current_trick_cards[state.current_player] = action
        new_state.player_hands[state.current_player].remove(action)

        if all(card is not None for card in new_state.current_trick_cards):
            # This trick is now finished
            # Score it
            new_state.prev_trick_scores = Hearts.get_trick_scores(new_state)
            new_state.player_scores += new_state.prev_trick_scores

            # Add it to the history
            new_state.past_trick_cards.extend(new_state.current_trick_cards)

            # Set trick leader to winner of previous trick
            new_state.current_trick_leader = Hearts.get_trick_winner(new_state)
            new_state.current_player = new_state.current_trick_leader

            # Intialize trick to Nones
            new_state.current_trick_cards = [None for _ in range(len(state.current_trick_cards))]
        else:
            # Continuing existing trick, set next player
            new_state.current_player +=  1
            new_state.current_player %=len(state.player_scores)

        return new_state

    @staticmethod
    def get_observable_state(state : HeartsState, player_index : int):
        return PartiallyObservedHeartsState(state.player_hands[player_index],
                                            state.current_trick_cards,
                                            state.current_trick_leader,
                                            len(state.past_trick_cards) == 0,
                                            any(card.suit == Suit.HEARTS for card in state.past_trick_cards))

    @staticmethod
    def get_trick_winner(state : HeartsState):
        led_card = state.current_trick_cards[state.current_trick_leader]
        winning_card = led_card
        winner = state.current_trick_leader
        for i, card in enumerate(state.current_trick_cards):
            if card.value > winning_card.value and card.suit == winning_card.suit:
                winner = i
                winning_card = card

        return winner

    @staticmethod
    def get_trick_scores(state : HeartsState):
        scores = np.zeros(len(state.current_trick_cards))
        trick_winner = Hearts.get_trick_winner(state)
        current_trick_points = (sum(1 for card in state.current_trick_cards if card.suit == Suit.HEARTS )
                                + (13 if Card(Value.QUEEN, Suit.SPADES) in state.current_trick_cards else 0))
        scores[trick_winner] = -1 * current_trick_points

        # Check if someone shot the moon, which could only be the trick winner
        new_scores = state.player_scores + scores
        if current_trick_points != 0 and new_scores[trick_winner] == -26:
            # The scores will set the trick winner to 0 and everyone else to 26
            scores = -26 * np.ones(len(state.current_trick_cards))
            scores[trick_winner] = -1 * state.player_scores[trick_winner]

        return scores

    @staticmethod
    def reward(state : HeartsState, player_index):
        if all(card is None for card in state.current_trick_cards):
            return state.prev_trick_scores[player_index]
        else:
            return 0

    @staticmethod
    def is_terminal_state(state : HeartsState):
        return all(len(hand) == 0 for hand in state.player_hands)

    @staticmethod
    def get_current_winning_card(state : PartiallyObservedHeartsState):
        led_card = state.current_trick_cards[state.current_trick_leader]
        winning_card = led_card
        winner = state.current_trick_leader
        for i, card in enumerate(state.current_trick_cards):
            if card is not None and card.value > winning_card.value and card.suit == winning_card.suit:
                winner = i
                winning_card = card

        return winning_card

    def __str__(self):
        return '\n'.join(
            [
                'Current Trick Cards: {}'.format(', '.join(map(str, self.state.current_trick_cards))),
                'Trick Led By: {}'.format(self.state.current_trick_leader),
                'Current Player: {}'.format(self.state.current_player),
                'Current Scores: {}'.format(self.state.player_scores),
                *['Player {} Cards: {}'.format(i, ', '.join(map(str,hand))) for i, hand in enumerate(self.state.player_hands)],
                'Legal Cards: {}'.format(', '.join(map(str, self.legal_actions(self.get_observable_state(self.state, self.state.current_player)))))
            ]
        )


class SimpleHeartsAgent(SequentialAgent):
    def __init__(self):
        super().__init__(game = Hearts)

    def choose_action(self, state : PartiallyObservedHeartsState, player_index, verbose = False):
        # Play the highest card that won't win the trick, otherwise play the lowest card
        legal_actions = self.game.legal_actions(state)

        current_winning_card = Hearts.get_current_winning_card(state)
        if current_winning_card is None:
            # Leading the trick, play lowest card
            return legal_actions[np.argmin(card.value for card in legal_actions)]

        highest_non_winning_card = None
        lowest_winning_card = None
        for card in legal_actions:
            if card.suit != current_winning_card.suit or card.value < current_winning_card.value:  # non-winning card
                if highest_non_winning_card is None or card.value > highest_non_winning_card.value:  # higher than current highest
                    highest_non_winning_card = card
            elif lowest_winning_card is None or card.value < lowest_winning_card.value:
                lowest_winning_card = card

        if highest_non_winning_card is not None:
            return highest_non_winning_card
        else:
            return lowest_winning_card

    def reward(self, reward_value, state, player_index):
        pass
