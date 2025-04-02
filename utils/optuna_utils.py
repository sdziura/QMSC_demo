import optuna
from config import OptunaParams
from training.train import Trainer


def objective(trial, trainer: Trainer) -> float:
    """
    Defines the objective function for Optuna hyperparameter optimization.

    Parameters
    ----------
    trial : optuna.trial.Trial
        A trial object that suggests hyperparameters.
    trainer : Trainer
        An instance of the Trainer class to train the model.

    Returns
    -------
    float
        The average validation loss for the suggested hyperparameters.
    """
    optuna_params = OptunaParams(
        # learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
        hidden_size_1=trial.suggest_categorical("hidden_size_1", [16, 32, 64, 128]),
        hidden_size_2=trial.suggest_categorical("hidden_size_2", [16, 32, 64, 128]),
        hidden_size_3=trial.suggest_categorical("hidden_size_3", [16, 32, 64, 128]),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        # dropout=trial.suggest_uniform("dropout", 0.1, 0.5),
    )

    # Pass the trial number to the train method
    return trainer.train(optuna_params, trial_number=trial.number)


def optimize_hyperparameters(trainer: Trainer, n_trials: int) -> dict:
    """
    Optimizes hyperparameters using Optuna and returns the best parameters.

    Parameters
    ----------
    trainer : Trainer
        An instance of the Trainer class to train the model.
    n_trials : int
        The number of trials for Optuna optimization.

    Returns
    -------
    dict
        The best hyperparameters found by Optuna.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, trainer), n_trials=n_trials)
    return study.best_params
