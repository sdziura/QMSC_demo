import logging
from sklearn.datasets import load_diabetes

from config import FixedParams, NNParams, QNNParams, QSVMParams, SVMParams
from evaluation import compare, t_student
from models.model_NN import TwoLayerModel
from models.model_QNN import VariationalQuantumCircuit
from models.model_QSVM import QSVM
from models.model_SVM import SVM
from training.train import Trainer
from utils.data_loader import load_data
from utils.optuna_utils import (
    optimize_hyperparameters_QNN,
    optimize_hyperparameters_QSVM,
    optimize_hyperparameters_SVM,
)
from utils.parameters_loader import load_params, save_params, save_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fusion_1():
    fixed_params = FixedParams(experiment_name="Fusion_1")

    qnn_params_dict = load_params("QNN")
    qnn_params = QNNParams(**qnn_params_dict)
    model_qnn = VariationalQuantumCircuit(
        fixed_params=fixed_params, QNN_params=qnn_params
    )
    trainer = Trainer(fixed_params=fixed_params)
    results = trainer.train(model=model_qnn)

    save_results(results, "qnn_vs_nn_1.json")
