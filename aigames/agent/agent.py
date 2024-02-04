import torch


class Agent:
    def get_action(self, state, legal_actions) -> int:
        raise NotImplementedError()

    def on_reward(self, reward, next_state, player_index):
        pass

    def before_game_start(self, n_players):
        pass

    def on_game_end(self):
        pass

    def on_action(self, state, action, next_state):
        pass


class AgentMulti:
    def get_actions(self, states: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        raise NotImplementedError()

    def on_rewards(self, rewards: torch.Tensor, mask: torch.Tensor):
        pass

    def before_game_start(self, game):
        pass

    def on_game_end(self):
        pass

    def on_action(self, state, action, next_state):
        pass
