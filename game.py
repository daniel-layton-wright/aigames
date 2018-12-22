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
                player.reward(self.reward(self.state, i), self.state, i)

                if self.verbose:
                    print(self)

                next_action = player.choose_action(self.state, i)
                while next_action not in self.legal_actions(self.state):
                    player.reward(self.ILLEGAL_ACTION_PENALTY, self.state, i)
                    next_action = player.choose_action(self.state, i)

                self.state = self.get_next_state(self.state, next_action)

                if self.is_terminal_state(self.state):
                    break

        for i, player in enumerate(self.players):
            player.reward(self.reward(self.state, i), self.state, i)

        if self.verbose:
            print(self)
            print('Game over.')