import logging
import mlflow
import numpy as np
from sklearn.model_selection import StratifiedKFold
from training.train import Trainer
from utils.data_loader import load_data
from utils.mlflow_utils import (
    initialize_mlflow,
    log_mlflow_params,
    log_params_with_prefix,
)
from evaluation.t_student import compute_corrected_ttest


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare(model1, model2, fixed_params) -> dict:
    trainer = Trainer(fixed_params)
    skf = StratifiedKFold(
        n_splits=fixed_params.folds,
        shuffle=True,
        random_state=fixed_params.random_state,
    )

    val_losses_1 = []
    val_f1s_1 = []

    val_losses_2 = []
    val_f1s_2 = []

    initialize_mlflow(fixed_params.mlflow_uri, fixed_params.experiment_name)
    run_name = f"CrossValidation_Experiment_{model1.model_params.model_type}_Compare_{model2.model_params.model_type}"

    with mlflow.start_run(run_name=run_name):
        log_mlflow_params(fixed_params.__dict__)
        log_params_with_prefix(model1.model_params.__dict__)
        log_params_with_prefix(model2.model_params.__dict__)
        X, y = load_data(fixed_params.dataset_file)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

            logger.info(f"Fold {fold+1}")

            with mlflow.start_run(
                nested=True, run_name=f"Fold_{fold+1}_{model1.model_params.model_type}"
            ):
                val_loss_1, val_f1_1 = trainer.train_fold_dispatch[
                    model1.model_params.model_type
                ](fold, X, y, train_idx, val_idx, 0, model1.model_params)

            with mlflow.start_run(
                nested=True, run_name=f"Fold_{fold+1}_{model2.model_params.model_type}"
            ):
                val_loss_2, val_f1_2 = trainer.train_fold_dispatch[
                    model2.model_params.model_type
                ](fold, X, y, train_idx, val_idx, 0, model2.model_params)

            val_losses_1.append(val_loss_1)
            val_f1s_1.append(val_f1_1)

            val_losses_2.append(val_loss_2)
            val_f1s_2.append(val_f1_2)

    f1_diff = [a - b for a, b in zip(val_f1s_1, val_f1s_2)]
    t, p = compute_corrected_ttest(f1_diff, len(train_idx), len(val_idx))

    return {
        "t": t,
        "p": p,
        "model_type_1": model1.model_params,
        "f1_model_1": val_f1s_1,
        "model_type_2": model2.model_params,
        "f1_model_2": val_f1s_2,
        "differences": f1_diff,
    }
