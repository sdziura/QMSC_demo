import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import mlflow

import logging
from config import FixedParams, ModelParams, NNParams, SVMParams, QNNParams, QSVMParams
from models.model_NN import TwoLayerModel
from models.model_SVM import SVM
from models.model_QSVM import QSVM
from models.model_QNN import VariationalQuantumCircuit
from utils.data_loader import load_data, get_dataloader
from utils.mlflow_utils import log_mlflow_params, initialize_mlflow
from training.trainer import get_trainer

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

    def __init__(self, fixed_params: FixedParams = FixedParams()):
        """
        Initializes the Trainer class, sets up MLFlow tracking, and loads data.
        """
        self.fixed_params = fixed_params

        self.train_fold_dispatch = {
            "svm": self.train_fold_SVM,
            "nn": self.train_fold_NN,
            "qnn": self.train_fold_NN,
            "qsvm": self.train_fold_SVM,
        }

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
        X, y = load_data(self.fixed_params.dataset_file)

        skf = StratifiedKFold(
            n_splits=self.fixed_params.folds,
            shuffle=True,
            random_state=self.fixed_params.random_state,
        )

        val_losses = []
        val_accs = []
        val_f1s = []

        # Start MLFlow experiment
        initialize_mlflow(
            self.fixed_params.mlflow_uri, self.fixed_params.experiment_name
        )

        with mlflow.start_run(run_name=run_name):
            log_mlflow_params(self.fixed_params.__dict__)
            log_mlflow_params(model_params.__dict__)
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                with mlflow.start_run(nested=True, run_name=f"Fold_{fold+1}"):
                    logger.info(f"Fold {fold+1}")
                    val_loss, val_f1 = self.train_fold_dispatch[model_type](
                        fold, X, y, train_idx, val_idx, trial_number, model_params
                    )
                    mlflow.log_metric("val_f1", val_f1)
                    mlflow.log_metric("val_loss", val_loss)
                    val_losses.append(val_loss)
                    val_f1s.append(val_f1)

            mean_val_loss = np.mean(val_losses)
            std_val_loss = np.std(val_losses)

            mean_val_acc = np.mean(val_accs)
            std_val_acc = np.std(val_accs)

            mean_val_f1 = np.mean(val_f1s)
            std_val_f1 = np.std(val_f1s)

            # Log aggregated results
            mlflow.log_metric("val_loss", mean_val_loss)
            mlflow.log_metric("val_loss_std", std_val_loss)
            mlflow.log_metric("val_acc", mean_val_acc)
            mlflow.log_metric("val_acc_std", std_val_acc)
            mlflow.log_metric("val_f1", mean_val_f1)
            mlflow.log_metric("val_f1_std", std_val_f1)

        return mean_val_loss, mean_val_f1

    def train_fold_NN(
        self,
        fold: int,
        X: torch.tensor,
        y: torch.tensor,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        trial_number: int,
        model_params: ModelParams,
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
        model_params : ModelParams
            An instance of ModelParams containing hyperparameters to be optimized.

        Returns
        -------
        tuple[float, float]
            The validation loss and validation accuracy for the fold.
        """
        dataset = TensorDataset(X, y)
        train_loader = get_dataloader(
            dataset=dataset,
            indices=train_idx,
            shuffle=True,
            batch_size=model_params.batch_size,
        )
        val_loader = get_dataloader(
            dataset=dataset,
            indices=val_idx,
            shuffle=False,
            batch_size=model_params.batch_size,
        )

        if model_params.model_type == "nn":
            model = TwoLayerModel(
                fixed_params=self.fixed_params, NN_params=model_params
            )
        elif model_params.model_type == "qnn":
            model = VariationalQuantumCircuit(
                fixed_params=self.fixed_params, QNN_params=model_params
            )
        else:
            raise ValueError("Invalid model type given")

        if FixedParams.use_gpu:
            model = model.to("cuda")
        logger.info(f"Model is running on device: {next(model.parameters()).device}")

        tb_run_name = (
            f"{self.fixed_params.experiment_name}/Trial_{trial_number}/Fold_{fold+1}"
        )
        trainer = get_trainer(self.fixed_params, tb_run_name)

        trainer.fit(model, train_loader, val_loader)

        val_loss = trainer.callback_metrics["val_loss"].item()
        val_acc = trainer.callback_metrics["val_acc"].item()
        val_f1 = trainer.callback_metrics["val_f1"].item()

        return val_loss, val_f1

    def train_fold_SVM(
        self,
        fold: int,
        X: torch.tensor,
        y: torch.tensor,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        trial_number: int,
        model_params: ModelParams,
    ) -> tuple[float, float]:
        """
        Trains an SVM model for a specific fold.

        Returns
        -------
        tuple[float, float]
            The validation loss (0 for SVM) and validation F1 score for the fold.
        """
        if model_params.model_type == "svm":
            model = SVM(fixed_params=self.fixed_params, SVM_params=model_params)
        elif model_params.model_type == "qsvm":
            model = QSVM(fixed_params=self.fixed_params, QSVM_params=model_params)
        else:
            raise ValueError("Invalid model type given")

        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])

        val_f1 = f1_score(y[val_idx].cpu(), y_pred.cpu(), average="weighted")

        return 0, val_f1
