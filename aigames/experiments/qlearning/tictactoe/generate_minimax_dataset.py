from aigames.agent.minimax_agent import MinimaxAgent
from aigames.agent.qlearning_agent import QLearningDataListener
from typing import Type
from aigames.game import PartiallyObservableSequentialGame, SequentialGame
from aigames.game.tictactoe import TicTacToe
from aigames.training_manager.qlearning_training_manager import BasicQLearningDataset
import copy
from tqdm.auto import tqdm
import pickle


class MinimaxAgentQData(MinimaxAgent):
    """
    A class to collect Qlearning-style data from minimax agent
    """

    def __init__(self, game: Type[SequentialGame], player_index,
                 data_listener: QLearningDataListener, discount_rate):
        super().__init__(game, discount_rate)
        self.game = game
        self.all_actions = game.get_all_actions()
        self.player_index = player_index
        self.data_listener = data_listener
        self.discount_rate = discount_rate
        self.last_state = None
        self.last_action_index = None
        self.cum_reward = 0
        self.training = True

    def get_action(self, state, legal_actions) -> int:
        self.last_state = copy.deepcopy(state)
        legal_action_indices = [self.all_actions.index(legal_action) for legal_action in legal_actions]

        action_idx = super().get_action(state, legal_actions)

        self.last_action_index = legal_action_indices[action_idx]
        return action_idx

    def on_reward(self, reward, next_state, player_index):
        if self.game.get_cur_player_index(next_state) == self.player_index or self.game.is_terminal_state(next_state):
            # It's back to this player's turn (or it's the end of the game)
            self.cum_reward += reward

            if self.last_state is not None and self.training:
                # Call back to the data listener on a data point (if we are traini  ng)
                self.data_listener.on_SARS(self.last_state, self.last_action_index, self.cum_reward, copy.deepcopy(next_state))

            # Reset the cum_reward
            self.cum_reward = 0
        else:
            # It's someone else's turn let's just keep track of the reward
            self.cum_reward += reward


def main():
    dataset0 = BasicQLearningDataset(TicTacToe, (1, 3, 3))
    dataset1 = BasicQLearningDataset(TicTacToe, (1, 3, 3))
    agent0 = MinimaxAgentQData(TicTacToe, 0, dataset0, discount_rate=0.99)
    agent1 = MinimaxAgentQData(TicTacToe, 1, dataset1, discount_rate=0.99)

    n_games = 2000
    for _ in tqdm(range(n_games)):
        game = TicTacToe([agent0, agent1])
        game.play()

    with open('dataset0.pkl', 'wb') as f:
        pickle.dump(dataset0, f)

    with open('dataset1.pkl', 'wb') as f:
        pickle.dump(dataset1, f)


if __name__ == '__main__':
    main()
