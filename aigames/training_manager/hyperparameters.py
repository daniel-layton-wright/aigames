from dataclasses import dataclass, field
from aigames.training_manager.alpha_dataset_multi import AlphaDatasetMultiHyperparameters


@dataclass(kw_only=True, slots=True)
class AlphaMultiTrainingHyperparameters(AlphaDatasetMultiHyperparameters):
    n_parallel_games: int = 1000
    value_weight_in_loss: float = 1.0
    game_listeners: list = field(default_factory=list)
    lr: float = 0.01
    weight_decay: float = 1e-5
    self_play_every_n_epochs: int = 10
    n_parallel_eval_games: int = 100
    eval_game_every_n_epochs: int = 10  # Set to -1 for never
    eval_game_network_only_every_n_epochs: int = 10
    eval_game_on_start: bool = False  # If True, run the full evaluation game before the training starts
    eval_game_listeners: list = field(default_factory=list)
    dataset_class: str = 'aigames.training_manager.alpha_dataset_multi.TrajectoryDataset'
    clear_dataset_before_self_play_rounds: list = field(default_factory=list)
    save_dataset_in_checkpoint: bool = False
    network_class: str = ''
    network_args: dict = field(default_factory=dict)
