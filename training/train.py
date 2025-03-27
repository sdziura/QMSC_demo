import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
import mlflow
import optuna
from pytorch_lightning.callbacks import EarlyStopping

import logging
from config import FixedParams, OptunaParams
from models.model import TwoLayerModel
from utils.data_loader import load_data, get_dataloader
from utils.mlflow_utils import log_mlflow_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Train:
    """
    A class used to train a machine learning model using PyTorch Lightning and MLFlow for experiment tracking.

    Attributes
    ----------
    fixed_params : FixedParams
        An instance of FixedParams containing fixed hyperparameters for training.
    X : torch.Tensor
        The input data tensor.
    y : torch.Tensor
        The labels tensor.

    Methods
    -------
    __init__():
        Initializes the Train class, sets up MLFlow tracking, and loads data.
    train(optuna_params: OptunaParams) -> float:
        Trains the model using cross-validation.
    train_fold(fold: int, train_loader: DataLoader, val_loader: DataLoader, optuna_params: OptunaParams) -> float:
        Trains the model for a specific fold and logs the results.
    objective(trial) -> float:
        Defines the objective function for Optuna hyperparameter optimization.
    optimize_hyperparameters() -> dict:
        Optimizes hyperparameters using Optuna and returns the best parameters.
    """

    def __init__(self):
        """
        Initializes the Train class, sets up MLFlow tracking, and loads data.
        """
        self.fixed_params = FixedParams()
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        mlflow.set_experiment(self.fixed_params.experiment_name)
        mlflow.pytorch.autolog()

        self.X, self.y = load_data(self.fixed_params.dataset_file)

    def train(self, optuna_params: OptunaParams, trial_number: int) -> float:
        """
        Trains the model using cross-validation.

        Parameters
        ----------
        optuna_params : OptunaParams
            An instance of OptunaParams containing hyperparameters to be optimized.
        trial_number : int
            The Optuna trial number.

        Returns
        -------
        float
            The average validation loss across all folds.
        """
        run_name = f"CrossValidation_Experiment_Trial_{trial_number}"
        mlflow.start_run(run_name=run_name)
        dataset = TensorDataset(self.X, self.y)
        skf = StratifiedKFold(
            n_splits=self.fixed_params.folds,
            shuffle=True,
            random_state=self.fixed_params.random_state,
        )

        val_losses = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            logger.info(f"Fold {fold+1}")

            train_loader = get_dataloader(
                dataset=dataset,
                indices=train_idx,
                shuffle=True,
                batch_size=optuna_params.batch_size,
            )
            val_loader = get_dataloader(
                dataset=dataset,
                indices=val_idx,
                shuffle=False,
                batch_size=optuna_params.batch_size,
            )

            val_loss = self.train_fold(fold, train_loader, val_loader, optuna_params)
            val_losses.append(val_loss)

        return np.mean(val_losses)

    def train_fold(
        self,
        fold: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optuna_params: OptunaParams,
    ) -> float:
        """
        Trains the model for a specific fold and logs the results.

        Parameters
        ----------
        fold : int
            The fold number.
        train_loader : DataLoader
            The DataLoader for the training data.
        val_loader : DataLoader
            The DataLoader for the validation data.
        optuna_params : OptunaParams
            An instance of OptunaParams containing hyperparameters to be optimized.

        Returns
        -------
        float
            The validation loss for the fold.
        """
        model = TwoLayerModel(
            fixed_params=self.fixed_params, optuna_params=optuna_params
        )

        tb_logger = TensorBoardLogger("logs/", name="my_model")

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, verbose=True, mode="min"
        )

        with mlflow.start_run(nested=True, run_name=f"Fold_{fold+1}"):
            log_mlflow_params(self.fixed_params.__dict__)
            log_mlflow_params(optuna_params.__dict__)
            trainer = pl.Trainer(
                max_epochs=self.fixed_params.max_epochs,
                accelerator="auto",
                val_check_interval=self.fixed_params.val_check_interval,
                log_every_n_steps=self.fixed_params.log_every_n_steps,
                logger=tb_logger,
                callbacks=[
                    pl.callbacks.ModelCheckpoint(
                        dirpath="checkpoints",
                        filename="best_model",
                        save_top_k=1,
                        monitor="val_loss",
                        mode="min",
                    ),
                    early_stopping,
                ],
            )

            trainer.fit(model, train_loader, val_loader)
            val_loss = trainer.callback_metrics["val_loss"].item()

        return val_loss

    def objective(self, trial) -> float:
        """
        Defines the objective function for Optuna hyperparameter optimization.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object that suggests hyperparameters.

        Returns
        -------
        float
            The average validation loss for the suggested hyperparameters.
        """
        optuna_params = OptunaParams(
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
            hidden_size_1=trial.suggest_categorical("hidden_size_1", [16, 32, 64, 128]),
            hidden_size_2=trial.suggest_categorical("hidden_size_2", [16, 32, 64, 128]),
            hidden_size_3=trial.suggest_categorical("hidden_size_3", [16, 32, 64, 128]),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            dropout=trial.suggest_uniform("dropout", 0.1, 0.5),
        )

        # Pass the trial number to the train method
        return self.train(optuna_params, trial_number=trial.number)

    def optimize_hyperparameters(self) -> dict:
        """
        Optimizes hyperparameters using Optuna and returns the best parameters.

        Returns
        -------
        dict
            The best hyperparameters found by Optuna.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.fixed_params.optuna_trials)
        return study.best_params
