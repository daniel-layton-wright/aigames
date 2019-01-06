import time
import tqdm


class Game:
    @staticmethod
    def reward(state, player_index):
        raise NotImplementedError()

    @staticmethod
    def legal_actions(state):
        raise NotImplementedError()

    @staticmethod
    def is_terminal_state(state):
        raise NotImplementedError()


class SequentialGame(Game):
    @staticmethod
    def get_next_state(state, action):
        raise NotImplementedError()

    def play(self):
        while not self.is_terminal_state(self.state):
            for i, player in enumerate(self.players):
                # i, player correspond to the player about to move

                if self.verbose:
                    print(self)
                    time.sleep(self.pause_seconds)

                if self.debugger:
                    self.debugger(self)

                next_action = player.choose_action(self.state, i, verbose = self.verbose)
                while next_action not in self.legal_actions(self.state):
                    player.reward(self.ILLEGAL_ACTION_PENALTY, self.state, i)
                    next_action = player.choose_action(self.state, i)

                self.state = self.get_next_state(self.state, next_action)

                for j, p in enumerate(self.players):
                    # j, p are just to issue rewards after each state
                    p.reward(self.reward(self.state, j), self.state, j)

                if self.is_terminal_state(self.state):
                    break

        if self.verbose:
            print(self)
            print('Game over.')