import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
import mlflow
from mlflow.models import infer_signature

from model import TwoLayerModel


class Train:
    def __init__(self):
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("HMM_Classification")
        mlflow.pytorch.autolog()

        self.params = {
            "max_iter": 1,
            "folds": 1,
            "random_state": 42,
            "batch_size": 32,
            "val_check_interval": 1,
            "log_every_n_steps": 10,
        }
        self.load_data()

    def load_data(self, file_path="hmm_gaussian_chains.h5"):
        with h5py.File(file_path, "r") as f:
            labels = f["label"][:]
            observations = f["observed"][:]

        # Flatten each (10,2) sequence into a 1D vector of size 20
        observations = observations.reshape(observations.shape[0], -1)
        self.X = torch.tensor(observations, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

        print("Observations shape:", self.X.shape)
        print("Labels shape:", self.y.shape)

    def train(self):
        # Convert dataset into a list of indices for cross-validation
        dataset = TensorDataset(self.X, self.y)
        skf = StratifiedKFold(
            n_splits=self.params["folds"],
            shuffle=True,
            random_state=self.params["random_state"],
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"\nFold {fold+1}")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # Create a PyTorch Dataset & DataLoader
            train_loader = DataLoader(
                train_subset,
                batch_size=self.params["batch_size"],
                shuffle=True,
                num_workers=4,
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.params["batch_size"],
                shuffle=False,
                num_workers=4,
            )

            model = TwoLayerModel(input_size=self.X.shape[1])

            # Define a TensorBoard logger
            tb_logger = TensorBoardLogger("logs/", name="my_model")

            # Start MLFlow run
            with mlflow.start_run(run_name=f"Fold_{fold+1}"):
                mlflow.log_params(self.params)  # Log hyperparameters
                trainer = pl.Trainer(
                    max_epochs=self.params["max_iter"],
                    accelerator="auto",
                    val_check_interval=self.params["val_check_interval"],
                    log_every_n_steps=self.params["log_every_n_steps"],
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
                input_example = torch.randn(1, self.X.shape[1])
                input_exampe_np = input_example.numpy()
                signature = infer_signature(
                    input_exampe_np, model(input_example).detach().numpy()
                )
                mlflow.pytorch.log_model(
                    model,
                    f"model_fold_{fold+1}",
                    input_example=input_exampe_np,
                    signature=signature,
                )


if __name__ == "__main__":
    model = Train()
    model.train()
