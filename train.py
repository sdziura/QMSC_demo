import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
import mlflow
from mlflow.models import infer_signature
import optuna

import logging
from params import FixedParams, OptunaParams
from model import TwoLayerModel

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
    get_dataloader(dataset: TensorDataset, indices: np.ndarray, shuffle: bool) -> DataLoader:
        Creates a DataLoader for a given dataset and indices.
    """

    def __init__(self):
        mlflow.set_tracking_uri(
            uri="http://127.0.0.1:8080"
        )  # command to start server locally: mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

        mlflow.set_experiment("HMM_Classification_2")
        mlflow.pytorch.autolog()

        self.fixed_params = FixedParams()
        self.load_data()

    def load_data(self, file_path: str = "hmm_gaussian_chains.h5") -> None:
        with h5py.File(file_path, "r") as f:
            labels, observations = f["label"][:], f["observed"][:]

        # Flatten each (10,2) sequence into a 1D vector of size 20
        observations = observations.reshape(observations.shape[0], -1)
        self.X = torch.tensor(observations, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

        logger.info(f"Observations shape: {self.X.shape}")
        logger.info(f"Labels shape: {self.y.shape}")

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

            # Create a PyTorch Dataset & DataLoader
            train_loader = self.get_dataloader(
                dataset=dataset, indices=train_idx, shuffle=True
            )
            val_loader = self.get_dataloader(
                dataset=dataset, indices=val_idx, shuffle=False
            )

            val_loss = self.train_fold(fold, train_loader, val_loader, optuna_params)
            val_losses.append(val_loss)

        return np.mean(val_losses)

    def train_fold(
        self, fold: int, train_loader: DataLoader, val_loader: DataLoader, optuna_params: OptunaParams
    ) -> float:
        model = TwoLayerModel(
            fixed_params=self.fixed_params, optuna_params=optuna_params
        )

        # Define a TensorBoard logger
        tb_logger = TensorBoardLogger("logs/", name="my_model")

        # Start MLFlow run
        with mlflow.start_run(run_name=f"Fold_{fold+1}"):
            mlflow.log_params(self.fixed_params.__dict__)  # Log fixed hyperparameters
            mlflow.log_params(optuna_params.__dict__)  # Log Optuna hyperparameters
            trainer = pl.Trainer(
                max_epochs=optuna_params.max_iter,
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
                    )
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

    def get_dataloader(
        self, dataset: TensorDataset, indices: np.ndarray, shuffle: bool
    ) -> DataLoader:
        subset = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=self.fixed_params.batch_size,
            shuffle=shuffle,
            num_workers=4,
        )

    def objective(self, trial):
        # Suggest hyperparameters for Optuna to optimize
        optuna_params = OptunaParams(
            max_iter=trial.suggest_int("max_iter", 1, 100),
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
            hidden_size_1=trial.suggest_int("hidden_size_1", 16, 128),
            hidden_size_2=trial.suggest_int("hidden_size_2", 16, 128),
            hidden_size_3=trial.suggest_int("hidden_size_3", 16, 128),
        )

        # Use the train method to train the model and return the average validation loss
        return self.train(optuna_params)

    def optimize_hyperparameters(self, n_trials: int = 100):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params


def main():
    trainer = Train()
    best_params = trainer.optimize_hyperparameters(n_trials=50)
    logger.info(f"Best hyperparameters: {best_params}")

    optuna_params = OptunaParams(**best_params)
    trainer.train(optuna_params)

    # Log the final model with the best hyperparameters
    best_model = TwoLayerModel(fixed_params=trainer.fixed_params, optuna_params=optuna_params)
    trainer.log_mlflow_model(best_model)


if __name__ == "__main__":
    main()
