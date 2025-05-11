import numpy as np
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import mlflow

import logging
from config import FixedParams, ModelParams, NNParams, SVMParams, QNNParams
from models.model_NN import TwoLayerModel
from models.model_SVM import SVM
from models.model_QNN import VariationalQuantumCircuit
from utils.data_loader import load_data, get_dataloader
from utils.mlflow_utils import log_mlflow_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
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
        Initializes the Trainer class, sets up MLFlow tracking, and loads data.
    train(optuna_params: OptunaParams, trial_number: int) -> float:
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
        Initializes the Trainer class, sets up MLFlow tracking, and loads data.
        """
        self.fixed_params = FixedParams()
        mlflow.set_tracking_uri(self.fixed_params.mlflow_uri)
        mlflow.set_experiment(self.fixed_params.experiment_name)
        mlflow.pytorch.autolog()
        self.train_fold_dispatch = {
            "svm": self.train_fold_SVM,
            "nn": self.train_fold_NN,
        }

        self.X, self.y = load_data(self.fixed_params.dataset_file)

    def train(
        self, model_type: str, model_params: ModelParams, trial_number: int = 0
    ) -> float:
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
        if model_type not in self.train_fold_dispatch:
            raise ValueError(f"Unknown model type: {model_type}")

        run_name = f"CrossValidation_Experiment_{model_type}_Trial_{trial_number}"
        skf = StratifiedKFold(
            n_splits=self.fixed_params.folds,
            shuffle=True,
            random_state=self.fixed_params.random_state,
        )

        val_losses = []
        val_accs = []
        with mlflow.start_run(run_name=run_name):
            log_mlflow_params(self.fixed_params.__dict__)
            log_mlflow_params(model_params.__dict__)
            for fold, (train_idx, val_idx) in enumerate(
                # skf.split(self.X.cpu(), self.y.cpu())
                skf.split(self.X, self.y)
            ):
                with mlflow.start_run(nested=True, run_name=f"Fold_{fold+1}"):
                    logger.info(f"Fold {fold+1}")
                    val_loss, val_acc = self.train_fold_dispatch[model_type](
                        fold, train_idx, val_idx, trial_number, model_params
                    )
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)

            mean_val_loss = np.mean(val_losses)
            std_val_loss = np.std(val_losses)

            mean_val_acc = np.mean(val_accs)
            std_val_acc = np.std(val_accs)

            # Log aggregated results
            mlflow.log_metric("val_loss", mean_val_loss)
            mlflow.log_metric("val_loss_std", std_val_loss)
            mlflow.log_metric("val_acc", mean_val_acc)
            mlflow.log_metric("val_acc_std", std_val_acc)

        return mean_val_loss, mean_val_acc

    def train_fold_NN(
        self,
        fold: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        trial_number: int,
        NN_params: NNParams,
    ) -> tuple[float, float]:
        """
        Trains the model for a specific fold and logs the results.

        Parameters
        ----------
        fold : int
            The fold number.
        train_idx : np.ndarray
            The indices for the training data.
        val_idx : np.ndarray
            The indices for the validation data.
        trial_number : int
            The Optuna trial number.
        optuna_params : OptunaParams
            An instance of OptunaParams containing hyperparameters to be optimized.

        Returns
        -------
        tuple[float, float]
            The validation loss and validation accuracy for the fold.
        """
        dataset = TensorDataset(self.X, self.y)
        train_loader = get_dataloader(
            dataset=dataset,
            indices=train_idx,
            shuffle=True,
            batch_size=NN_params.batch_size,
        )
        val_loader = get_dataloader(
            dataset=dataset,
            indices=val_idx,
            shuffle=False,
            batch_size=NN_params.batch_size,
        )

        model = TwoLayerModel(fixed_params=self.fixed_params, NN_params=NN_params)
        if FixedParams.use_gpu:
            model = model.to("cuda")
        logger.info(f"Model is running on device: {next(model.parameters()).device}")

        tb_run_name = (
            f"{self.fixed_params.experiment_name}/Trial_{trial_number}/Fold_{fold+1}"
        )
        tb_logger = TensorBoardLogger("server/tb_logs/", name=tb_run_name)

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, verbose=True, mode="min"
        )

        trainer = pl.Trainer(
            max_epochs=self.fixed_params.max_epochs,
            accelerator="gpu" if FixedParams.use_gpu else "auto",
            val_check_interval=self.fixed_params.val_check_interval,
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
        val_acc = trainer.callback_metrics["val_acc"].item()

        return val_loss, val_acc

    def train_fold_SVM(
        self,
        fold: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        trial_number: int,
        SVM_params: SVMParams,
    ) -> tuple[float, float]:

        model = SVM(fixed_params=self.fixed_params, SVM_params=SVM_params)

        # tb_run_name = (
        #     f"{self.fixed_params.experiment_name}/Trial_{trial_number}/Fold_{fold+1}"
        # )
        # tb_logger = TensorBoardLogger("server/tb_logs/", name=tb_run_name)

        model.fit(self.X[train_idx], self.y[train_idx])
        y_pred = model.predict(self.X[val_idx])

        val_acc = accuracy_score(self.y[val_idx], y_pred)

        return 0, val_acc
