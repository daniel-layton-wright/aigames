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


class GameListenerMulti:
    def before_game_start(self, game):
        pass

    def before_action(self, game, legal_actions):
        pass

    def on_action(self, game, action):
        pass

    def after_action(self, game):
        pass

    def on_game_end(self, game):
        pass

    def on_states_from_env(self, game):
        pass

    def on_rewards(self, rewards, mask):
        pass

    def on_game_restart(self, game):
        pass

    def before_env_move(self, states: torch.Tensor, mask: torch.Tensor):
        pass


class AgentMulti(GameListenerMulti):
    def get_actions(self, states: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        raise NotImplementedError()
