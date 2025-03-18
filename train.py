import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
import mlflow
from mlflow.models import infer_signature

import logging
from dataclasses import dataclass

from model import TwoLayerModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainParams:
    max_iter: int = 1
    folds: int = 2
    random_state: int = 42
    batch_size: int = 32
    val_check_interval: int = 1
    log_every_n_steps: int = 1


class Train:
    def __init__(self):
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("HMM_Classification")
        mlflow.pytorch.autolog()

        self.params = TrainParams()
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

    def train(self) -> None:
        # Convert dataset into a list of indices for cross-validation
        dataset = TensorDataset(self.X, self.y)
        skf = StratifiedKFold(
            n_splits=self.params.folds,
            shuffle=True,
            random_state=self.params.random_state,
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            logger.info(f"\nFold {fold+1}")

            # Create a PyTorch Dataset & DataLoader
            train_loader = self.get_dataloader(
                dataset=dataset, indices=train_idx, shuffle=True
            )
            val_loader = self.get_dataloader(
                dataset=dataset, indices=val_idx, shuffle=False
            )

            self.train_fold(fold, train_loader, val_loader)

    def train_fold(
        self, fold: int, train_loader: DataLoader, val_loader: DataLoader
    ) -> None:
        model = TwoLayerModel(input_size=self.X.shape[1])

        # Define a TensorBoard logger
        tb_logger = TensorBoardLogger("logs/", name="my_model")

        # Start MLFlow run
        with mlflow.start_run(run_name=f"Fold_{fold+1}"):
            mlflow.log_params(self.params.__dict__)  # Log hyperparameters
            trainer = pl.Trainer(
                max_epochs=self.params.max_iter,
                accelerator="auto",
                val_check_interval=self.params.val_check_interval,
                log_every_n_steps=self.params.log_every_n_steps,
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

            # Log model with input example
            self.log_mlflow_model(model=model, fold=fold)

    def log_mlflow_model(self, model: pl.LightningModule, fold: int) -> None:
        input_example = torch.randn(1, self.X.shape[1])
        input_example_np = input_example.numpy()
        signature = infer_signature(
            input_example_np, model(input_example).detach().numpy()
        )

        mlflow.pytorch.log_model(
            model,
            f"model_fold_{fold+1}",
            input_example=input_example_np,
            signature=signature,
        )

    def get_dataloader(
        self, dataset: TensorDataset, indices: np.ndarray, shuffle: bool
    ) -> DataLoader:
        subset = Subset(dataset, indices)
        return DataLoader(
            subset, batch_size=self.params.batch_size, shuffle=shuffle, num_workers=4
        )


if __name__ == "__main__":
    model = Train()
    model.train()
