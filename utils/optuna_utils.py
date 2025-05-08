import optuna
from config import OptunaParams
from training.train import Trainer


def optimize_hyperparameters_NN(trainer: Trainer, n_trials: int) -> dict:
    def objective(trial, trainer: Trainer) -> float:
        optuna_params = OptunaParams(
            # learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
            hidden_size_1=trial.suggest_categorical("hidden_size_1", [16, 32, 64, 128]),
            hidden_size_2=trial.suggest_categorical("hidden_size_2", [16, 32, 64, 128]),
            # hidden_size_3=trial.suggest_categorical("hidden_size_3", [16, 32, 64, 128]),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            # dropout=trial.suggest_uniform("dropout", 0.1, 0.5),
        )
        # Pass the trial number to the train method
        return trainer.train(optuna_params, trial_number=trial.number)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, trainer), n_trials=n_trials)
    return study.best_params
