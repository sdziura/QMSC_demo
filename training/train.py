import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
import mlflow
from mlflow.models import infer_signature
import optuna
from pytorch_lightning.callbacks import EarlyStopping

import logging
from config import FixedParams, OptunaParams
from models.model import TwoLayerModel
from utils.data_loader import load_data, get_dataloader

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
    load_data(file_path: str = "hmm_gaussian_chains.h5") -> None:
        Loads data from an HDF5 file and preprocesses it.
    train(optuna_params: OptunaParams) -> float:
        Trains the model using cross-validation.
    train_fold(fold: int, train_loader: DataLoader, val_loader: DataLoader, optuna_params: OptunaParams) -> float:
        Trains the model for a specific fold and logs the results.
    log_mlflow_model(model: pl.LightningModule) -> None:
        Logs the trained model to MLFlow.
    get_dataloader(dataset: TensorDataset, indices: np.ndarray, shuffle: bool, batch_size: int) -> DataLoader:
        Creates a DataLoader for a given dataset and indices.
    """

    def __init__(self):
        self.fixed_params = FixedParams()
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        mlflow.set_experiment(self.fixed_params.experiment_name)
        mlflow.pytorch.autolog()

        self.X, self.y = load_data(self.fixed_params.dataset_file)

    def train(self, optuna_params: OptunaParams) -> float:
        # Convert dataset into a list of indices for cross-validation
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
                dataset=dataset, indices=train_idx, shuffle=True, batch_size=optuna_params.batch_size
            )
            val_loader = get_dataloader(
                dataset=dataset, indices=val_idx, shuffle=False, batch_size=optuna_params.batch_size
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
        model = TwoLayerModel(
            fixed_params=self.fixed_params, optuna_params=optuna_params
        )

        tb_logger = TensorBoardLogger("logs/", name="my_model")
        
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            verbose=True,
            mode="min"
        )

        with mlflow.start_run(run_name=f"Fold_{fold+1}"):
            mlflow.log_params(self.fixed_params.__dict__)
            mlflow.log_params(optuna_params.__dict__)
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
                    early_stopping
                ],
            )

            trainer.fit(model, train_loader, val_loader)
            val_loss = trainer.callback_metrics["val_loss"].item()

        return val_loss

    def log_mlflow_model(self, model: pl.LightningModule) -> None:
        input_example = torch.randn(1, self.X.shape[1])
        input_example_np = input_example.numpy()
        signature = infer_signature(
            input_example_np, model(input_example).detach().numpy()
        )

        mlflow.pytorch.log_model(
            model,
            "best_model",
            input_example=input_example_np,
            signature=signature,
        )

    def objective(self, trial):
        # Suggest hyperparameters for Optuna to optimize
        optuna_params = OptunaParams(
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
            hidden_size_1=trial.suggest_int("hidden_size_1", 16, 128, step=16),
            hidden_size_2=trial.suggest_int("hidden_size_2", 16, 128, step=16),
            hidden_size_3=trial.suggest_int("hidden_size_3", 16, 128, step=16),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            dropout=trial.suggest_uniform("dropout", 0.1, 0.5)
        )

        return self.train(optuna_params)

    def optimize_hyperparameters(self, n_trials: int = 100):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params


def main():
    trainer = Train()
    best_params = trainer.optimize_hyperparameters(n_trials=5)
    logger.info(f"Best hyperparameters: {best_params}")

    optuna_params = OptunaParams(**best_params)
    trainer.train(optuna_params)

    # Log the final model with the best hyperparameters
    best_model = TwoLayerModel(
        fixed_params=trainer.fixed_params, optuna_params=optuna_params
    )
    trainer.log_mlflow_model(best_model)


if __name__ == "__main__":
    main()
