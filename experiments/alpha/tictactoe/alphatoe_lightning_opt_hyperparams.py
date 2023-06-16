import argparse
import pytorch_lightning.loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from .alphatoe_lightning import *
import optuna
from ..utils.custom_optuna_pruner import CustomPyTorchLightningPruningCallback
import wandb
from itertools import chain


def objective(trial: optuna.Trial, args):
    network = TicTacToeNetwork()
    evaluator = FastTicTacToeEvaluator(network)
    hyperparams = AlphaTrainingHyperparameters()
    hyperparams.min_data_size = 10
    hyperparams.max_data_size = trial.suggest_int('max_data_size', 1000, 10000)
    hyperparams.n_mcts = 100
    hyperparams.dirichlet_alpha = trial.suggest_float('dirichlet_alpha', 0.01, 0.5, log=True)
    hyperparams.dirichlet_epsilon = trial.suggest_float('dirichlet_epsilon', 0.2, 0.8)
    hyperparams.lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    hyperparams.weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
    hyperparams.training_tau = TrainingTau(trial.suggest_float('training_tau', 0.3, 3, log=True))
    hyperparams.c_puct = trial.suggest_float('c_puct', 0.1, 10, log=True)
    hyperparams.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    wandb_kwargs = dict(
        project='aigames2',
        group='tictactoe_lightning',
        reinit=False,
        settings=wandb.Settings(start_method="fork")
    )

    # So we can get the name for model checkpointing
    wandb.init(**wandb_kwargs)

    wandb_run = wandb.run.name or os.path.split(wandb.run.path)[-1]

    callbacks = [
         CustomPyTorchLightningPruningCallback(trial, monitor="avg_reward_against_minimax_ema",
                                               step_variable="current_epoch",
                                               before_prune=lambda: wandb.finish()),
         ModelCheckpoint(dirpath=os.path.join(args.ckpt_dir, wandb_run), save_top_k=1, mode='max',
                         monitor='avg_reward_against_minimax_ema'),
     ]

    trainer = pl.Trainer(reload_dataloaders_every_n_epochs=1, logger=pl.loggers.WandbLogger(**wandb_kwargs),
                         max_epochs=args.n_epochs, callbacks=callbacks)

    slots = chain.from_iterable(getattr(cls, '__slots__', []) for cls in type(hyperparams).__mro__)
    params = {s: getattr(hyperparams, s) for s in slots if hasattr(hyperparams, s)}
    wandb.config.update(params)

    training_run = AlphaTrainingRunLightning(FastTicTacToe, evaluator, hyperparams)
    trainer.fit(training_run)

    wandb.finish()

    return training_run.avg_reward_against_minimax_ema


def main():
    wandb.setup()
    wandb.require("service")

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--optuna_study', '-o', type=str, required=False, default=f'{os.getcwd()}/optuna_experiment.log')
    parser.add_argument('--n_epochs', type=int, required=False, default=50, help='Number of epochs to train for per trial')
    args, _ = parser.parse_known_args()

    from optuna.storages import JournalFileStorage, JournalStorage
    storage = JournalStorage(JournalFileStorage(args.optuna_study))
    study = optuna.create_study(
        direction='maximize', study_name='tictactoe_lightning', load_if_exists=True,
        storage=storage,
        pruner=optuna.pruners.PatientPruner(
            optuna.pruners.PercentilePruner(
                0.5,
                n_startup_trials=10,
                n_min_trials=10,
                n_warmup_steps=10,  # this number is in epochs
                interval_steps=5,
            ),
            patience=5,
            min_delta=0
        )
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, n_jobs=args.n_jobs)


if __name__ == '__main__':
    main()
