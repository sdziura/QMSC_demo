import optuna
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from config import NNParams, SVMParams
from training.train import Trainer


def optimize_hyperparameters_NN(trainer: Trainer, n_trials: int) -> dict:
    def objective(trial, trainer: Trainer) -> float:
        optuna_params = NNParams(
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
            hidden_size_1=trial.suggest_categorical("hidden_size_1", [16, 32, 64, 128]),
            hidden_size_2=trial.suggest_categorical("hidden_size_2", [16, 32, 64, 128]),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            dropout=trial.suggest_uniform("dropout", 0.1, 0.5),
        )
        val_loss, _ = trainer.train("nn", optuna_params, trial_number=trial.number)
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, trainer), n_trials=n_trials)
    return study.best_params


def optimize_hyperparameters_SVM(trainer: Trainer, n_trials: int) -> dict:
    def objective(trial, trainer: Trainer) -> float:
        kernel = trial.suggest_categorical(
            "kernel", ["rbf", "linear", "poly", "sigmoid"]
        )
        C = trial.suggest_loguniform("C", 1e-3, 1e3)
        if kernel in ["rbf", "poly", "sigmoid"]:
            gamma = trial.suggest_loguniform("gamma", 1e-4, 1e1)
        else:
            gamma = "scale"

        if kernel == "poly":
            degree = trial.suggest_int("degree", 2, 5)
        else:
            degree = 3  # default value, not used if not "poly"

        optuna_params = SVMParams(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
        )
        _, val_acc = trainer.train("svm", optuna_params, trial_number=trial.number)

        return val_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, trainer=trainer), n_trials=n_trials)
    return study.best_params
