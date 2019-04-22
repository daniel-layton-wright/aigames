from ..games import PartiallyObservableSequentialGame


class QLearningMonitor:
    def before_training_start(self, qlearning_agent):
        pass

    def on_game_end(self, qlearning_agent, iter_number: int):
        pass

    def before_player_action(self, game: PartiallyObservableSequentialGame):
        pass
