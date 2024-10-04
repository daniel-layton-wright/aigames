"""
Learn to play Hearts with AlphaZero
"""
from typing import Type, Union

import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from aigames.agent.alpha_agent_multi import AlphaAgentMulti, ConstantMCTSIters
from aigames.agent.hearts.simple_hearts_agent import SimpleHeartsAgent
from aigames.experiments.alpha.hearts.network_architectures import HeartsNetworkHyperparameters
from aigames.experiments.alpha.utils.callbacks import CheckpointMidGame
from aigames.game.game_multi import GameMulti
from aigames.game.hearts import get_hearts_game_class
from aigames.training_manager.alpha_dataset_multi import AlphaDatasetMulti
from aigames.utils.listeners import ActionCounterProgressBar, RewardListenerMulti
from ....training_manager.alpha_training_manager_multi_lightning import AlphaMultiTrainingRunLightning
from aigames.training_manager.hyperparameters import AlphaMultiTrainingHyperparameters, EvalGameConfig
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from ....utils.utils import import_string
import os
import torch
import copy


class HeartsTrainingRun(AlphaMultiTrainingRunLightning):
    def __init__(self, game_class: Type[GameMulti],
                 hyperparams: AlphaMultiTrainingHyperparameters, agent_class=AlphaAgentMulti,
                 dataset: Union[AlphaDatasetMulti, Type[AlphaDatasetMulti], None] = None):
        super().__init__(game_class, hyperparams, agent_class, dataset)

    def create_eval_games(self):
        eval_games = []
        
        # Create three types of evaluation games
        eval_configs = [
            EvalGameConfig(n_parallel_games=self.hyperparams.n_parallel_games, n_mcts_iters=-1, eval_every_n_epochs=1, eval_on_start=True, 
                           listeners=[RewardListenerMulti(1.), ActionCounterProgressBar(52, 'Full eval game')]),
            EvalGameConfig(n_parallel_games=self.hyperparams.n_parallel_games, n_mcts_iters=0, eval_every_n_epochs=1, eval_on_start=True, 
                           listeners=[RewardListenerMulti(1.), ActionCounterProgressBar(52, 'No search eval game')]),
            EvalGameConfig(n_parallel_games=self.hyperparams.n_parallel_games, n_mcts_iters=-1, eval_every_n_epochs=1, eval_on_start=True, 
                           listeners=[RewardListenerMulti(1.), ActionCounterProgressBar(52, 'Search vs no search eval game')])
        ]
        
        for i, eval_config in enumerate(eval_configs):            
            eval_game = self.game_class(eval_config.n_parallel_games, self.agent, listeners=eval_config.listeners)
            eval_game.players = [self.agent] + [SimpleHeartsAgent() for _ in range(3)]
            
            if i == 2:
                # Agent w/search against agents without
                agent_no_search = copy.copy(self.agent)
                agent_no_search.hyperparams = copy.deepcopy(self.agent.hyperparams)
                agent_no_search.hyperparams.n_mcts_iters = ConstantMCTSIters(0)
                eval_game.players = [self.agent] + [agent_no_search for _ in range(3)]
            
            eval_games.append((eval_game, eval_config))
        
        return eval_games

    def log_game_results(self, game, suffix):
        log_dict = {}

        undiscounted_reward_listener = next(filter(lambda x: isinstance(x, RewardListenerMulti) and x.discount == 1.,
                                                   game.listeners), None)

        if undiscounted_reward_listener is not None:
            log_dict[f'avg_reward/{suffix}'] = undiscounted_reward_listener.rewards[:, 0].mean().item()
            log_dict[f'avg_reward/{suffix}/opponent'] = undiscounted_reward_listener.rewards[:, 1:].mean().item()

        self.log_to_wandb(log_dict)

    def after_self_play_game(self):
        self.log_game_results(self.game, 'train')
    
    def after_eval_game(self, eval_game: GameMulti, eval_config: EvalGameConfig):
        """
        Log results after an evaluation game.

        :param eval_game: The evaluation game that was played
        :param eval_config: Configuration for the evaluation game
        """
        suffix = "eval_"
        if eval_config.n_mcts_iters == 0:
            suffix += "no_search"
        elif eval_config.n_mcts_iters == -1:
            suffix += "full_search"
        else:
            suffix += f"search_{eval_config.n_mcts_iters}"

        if isinstance(eval_game.players[1], SimpleHeartsAgent):
            suffix += "_vs_simple"
        elif isinstance(eval_game.players[1], AlphaAgentMulti):
            suffix += "_vs_no_search"

        self.log_game_results(eval_game, suffix)


@hydra.main(config_name='base.yaml', config_path='./config/', version_base='1.3.2')
def main(cfg: DictConfig):
    hyperparams: AlphaMultiTrainingHyperparameters = instantiate(cfg.training_hypers, _convert_='all')

    checkpoint_mid_game = CheckpointMidGame(save_every_n_moves=5)
    Hearts = get_hearts_game_class(hyperparams.device)

    if 'restore_ckpt_path' in cfg and cfg.restore_ckpt_path:
        training_run = HeartsTrainingRun.load_from_checkpoint(cfg.restore_ckpt_path, map_location='cpu')
        hyperparams = training_run.hyperparams

    # remove the CheckpointMidGame if it exists and add the current one
    for x in hyperparams.game_listeners:
        if isinstance(x, CheckpointMidGame):
            hyperparams.game_listeners.remove(x)

    hyperparams.game_listeners.append(checkpoint_mid_game)
    hyperparams.game_listeners.append(ActionCounterProgressBar(52, 'Train game'))
    hyperparams.game_listeners.append(RewardListenerMulti(1.))

    training_run = HeartsTrainingRun(Hearts, hyperparams, dataset=import_string(hyperparams.dataset_class))

    # Start wandb run
    wandb_kwargs = dict(
        project='aigames2',
        group='hearts',
        reinit=False,
    )

    if cfg.restore_wandb_run_id is not None:
        wandb_kwargs['id'] = cfg.restore_wandb_run_id
        wandb_kwargs['resume'] = True

    # So we can get the name for model checkpointing
    wandb.init(**wandb_kwargs)
    wandb_run = wandb.run.name or os.path.split(wandb.run.path)[-1]
    wandb.run.watch(training_run.network)

    torch.autograd.set_detect_anomaly(True)

    # Print out the cfg in a nice way
    print(OmegaConf.to_yaml(cfg))

    model_checkpoint = ModelCheckpoint(dirpath=os.path.join(cfg.ckpt_dir, wandb_run), save_last=True)
    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=1,
                         logger=pl_loggers.WandbLogger(**wandb_kwargs),
                         callbacks=[model_checkpoint, checkpoint_mid_game], log_every_n_steps=1,
                         max_epochs=cfg.max_epochs,
                         gradient_clip_val=cfg.gradient_clip_val
                         )
    trainer.fit(training_run, ckpt_path=cfg.get('restore_ckpt_path'))

    if cfg.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()


"""
2:50 for 100games/100mcts
"""


"""
Evaluations:
1. Game with naive networks but search against SimpleAgents
2. Game with trained networks and no search against SimpleAgents
3. Game with trained networks and search against SimpleAgents

Once search doesn't improve over no search, the networks are considered trained.
"""