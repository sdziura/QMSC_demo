import logging
from config import FixedParams, NNParams, QNNParams
from evaluation import compare
from models.model_NN import TwoLayerModel
from models.model_QNN import VariationalQuantumCircuit
from training.train import Trainer
from utils.optuna_utils import (
    optimize_hyperparameters_QNN,
    optimize_hyperparameters_QSVM,
    optimize_hyperparameters_SVM,
)
from utils.parameters_loader import load_params, save_params, save_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_qnn_nn():
    fixed_params = FixedParams()

    qnn_params_dict = load_params("QNN")
    qnn_params = QNNParams(**qnn_params_dict)
    model_qnn = VariationalQuantumCircuit(
        fixed_params=fixed_params, QNN_params=qnn_params
    )

    nn_params_dict = load_params("NN")
    nn_params = NNParams(**nn_params_dict)
    model_nn = TwoLayerModel(fixed_params=fixed_params, NN_params=nn_params)

    results = compare(model_qnn, model_nn, fixed_params)
    save_results(results, "qnn_vs_nn_1.json")


def find_best_qnn():

    trainer = Trainer()

    # Optimize hyperparameters
    best_params = optimize_hyperparameters_QNN(
        trainer, n_trials=trainer.fixed_params.optuna_trials
    )
    logger.info(f"Best hyperparameters: {best_params}")
    save_params(best_params, "QNN_best_params")


def find_best_qsvm():

    trainer = Trainer()

    # Optimize hyperparameters
    best_params = optimize_hyperparameters_QSVM(
        trainer, n_trials=trainer.fixed_params.optuna_trials
    )
    logger.info(f"Best hyperparameters: {best_params}")
    save_params(best_params, "QSVM_best_params")
