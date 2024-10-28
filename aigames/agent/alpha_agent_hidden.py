from aigames.game.game_multi import GameMulti
from .alpha_agent_multi import AlphaAgentHyperparametersMulti, AlphaAgentMulti
import torch
from dataclasses import dataclass
from typing import Any, List, Type


@dataclass(kw_only=True, slots=True)
class AlphaAgentHiddenHyperparameters(AlphaAgentHyperparametersMulti):
    """
    Hyperparameters for AlphaAgentHidden.

    :param n_samples: Number of samples to generate for each state
    """
    n_samples: int = 10


class AlphaAgentHidden(AlphaAgentMulti):
    """
    AlphaAgentHidden class for handling hidden information games.
    """

    class Sampler:
        """
        Abstract base class for samplers used in AlphaAgentHidden.
        """

        def sample(self, states: torch.Tensor, player_indices: torch.Tensor) -> torch.Tensor:
            """
            Sample a single full state for each hidden state.

            :param states: Tensor of hidden states
            :param player_indices: Tensor of player indices
            :return: Tensor of sampled full states
            """
            raise NotImplementedError()
        
        def sample_multiple(self, states: torch.Tensor, player_indices: torch.Tensor, n_samples: int) -> torch.Tensor:
            """
            Sample multiple full states for each hidden state.

            :param states: Tensor of hidden states
            :param player_indices: Tensor of player indices
            :param n_samples: Number of samples to generate for each state
            :return: Tensor of sampled full states
            """
            raise NotImplementedError()
    
    def __init__(self, game_class: Type[GameMulti], evaluator: Any, hyperparams: AlphaAgentHiddenHyperparameters, 
                 listeners: List[Any], sampler: Sampler):
        """
        Initialize AlphaAgentHidden.

        :param game_class: Class representing the game
        :param evaluator: Evaluator for the agent
        :param hyperparams: Hyperparameters for the agent
        :param listeners: List of listeners for the agent
        :param sampler: Sampler for generating full states from hidden states
        """
        super().__init__(game_class, evaluator, hyperparams, listeners)
        self.sampler = sampler
    
    def get_actions(self, states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Get actions for the given states.

        :param states: Tensor of game states
        :param masks: Tensor of masks for which games are still in progress
        :return: Tensor of selected actions
        """
        # Shape of samples will be n_states x n_samples x *state_shape      
        player_indices = self.game_class.get_cur_player_index(states)  
        samples = self.sampler.sample_multiple(states, player_indices.long(), self.hyperparams.n_samples)
        
        samples = samples.reshape(states.shape[0] * self.hyperparams.n_samples, *self.game_class.get_state_shape())
        
        n_mcts_iters, _ = self.hyperparams.n_mcts_iters.get_n_mcts_iters()
        self.setup_mcts(self.hyperparams, n_mcts_iters, samples)
        self.mcts.search_for_n_iters(n_mcts_iters)
        _, pi = self.action_selector.get_final_actions_and_pi(self.mcts, tau=1.0)
        
        # Reshape pi
        pi = pi.reshape(states.shape[0], self.hyperparams.n_samples, -1)
            
        # Average the policies
        policies = torch.mean(pi, dim=1)
        policies /= torch.sum(policies, dim=-1, keepdim=True)
        
        # Sample actions from the policies
        actions = torch.multinomial(policies, num_samples=1).squeeze(-1)
        
        return actions
