import logging
from sklearn.datasets import load_diabetes
import torch.nn as nn

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
    fixed_params = FixedParams(experiment_name="Fusion_1_to delete")

    qnn_params = QNNParams(
        model_name="VQC_fusion_1",
        learning_rate=0.001,
        batch_size=256,
        n_layers=2,
        n_qubits=10,
        embedding_axis="X",
        embedding_axis_2="Y",
        rot_axis_0="X",
        rot_axis_1="Y",
        shots=None,
        loss_func=nn.CrossEntropyLoss(),
    )

    trainer = Trainer(fixed_params=fixed_params)
    results = trainer.train(model_params=qnn_params)

    save_results(results, "fusion_1.json")


def fusion_2():
    fixed_params = FixedParams(experiment_name="Fusion_2")

    qnn_params = QNNParams(
        model_name="VQC_fusion_1",
        learning_rate=0.001,
        batch_size=256,
        n_layers=2,
        n_qubits=5,  # For AmplitudeEmbedding n qubits for 2^n features.
        embedding_version=2,
        embedding_axis="X",
        embedding_axis_2="Y",
        rot_axis_0="X",
        rot_axis_1="Y",
        shots=None,
        loss_func=nn.CrossEntropyLoss(),
    )

    trainer = Trainer(fixed_params=fixed_params)
    results = trainer.train(model_params=qnn_params)

    save_results(results, "fusion_2.json")


def fusion_3():
    fixed_params = FixedParams(experiment_name="Fusion_3")

    qnn_params = QNNParams(
        model_name="VQC_fusion_1",
        learning_rate=0.001,
        batch_size=256,
        n_layers=2,
        # For AmplitudeEmbedding n qubits for 2^n features.
        # Taking half of features seperatly, means we need 2*n for 2^n >= 10
        n_qubits=8,
        embedding_version=3,
        ansatz_version=2,
        embedding_axis="X",
        embedding_axis_2="Y",
        rot_axis_0="X",
        rot_axis_1="Y",
        shots=None,
        loss_func=nn.CrossEntropyLoss(),
    )

    trainer = Trainer(fixed_params=fixed_params)
    results = trainer.train(model_params=qnn_params)

    save_results(results, "fusion_3.json")


def fusion_4():
    fixed_params = FixedParams(experiment_name="Fusion_4")

    qnn_params = QNNParams(
        model_name="VQC_fusion_1",
        learning_rate=0.001,
        batch_size=256,
        n_layers=2,
        # For AmplitudeEmbedding n qubits for 2^n features.
        # Taking half of features seperatly, means we need 2*n for 2^n >= 10
        n_qubits=8,
        embedding_version=3,
        ansatz_version=3,
        rot_axis_0="X",
        rot_axis_1="X",
        shots=None,
        loss_func=nn.CrossEntropyLoss(),
    )

    trainer = Trainer(fixed_params=fixed_params)
    results = trainer.train(model_params=qnn_params)

    save_results(results, "fusion_4.json")


def fusion_5a():
    fixed_params = FixedParams(experiment_name="Fusion_5a")

    qnn_params = QNNParams(
        model_name="VQC_fusion_1",
        learning_rate=0.001,
        batch_size=256,
        n_layers=2,
        n_qubits=5,  # For AmplitudeEmbedding n qubits for 2^n features.
        ansatz_version=4,
        embedding_version=2,
        embedding_axis="X",
        embedding_axis_2="Y",
        rot_axis_0="X",
        rot_axis_1="Y",
        shots=None,
        loss_func=nn.CrossEntropyLoss(),
    )

    trainer = Trainer(fixed_params=fixed_params)
    results = trainer.train(model_params=qnn_params)

    save_results(results, "fusion_5.json")


def fusion_5b():
    fixed_params = FixedParams(experiment_name="Fusion_5a")

    qnn_params = QNNParams(
        model_name="VQC_fusion_1",
        learning_rate=0.001,
        batch_size=256,
        n_layers=2,
        n_qubits=5,  # For AmplitudeEmbedding n qubits for 2^n features.
        ansatz_version=4,
        embedding_version=2,
        embedding_axis="X",
        embedding_axis_2="Y",
        rot_axis_0="X",
        rot_axis_1="Y",
        shots=None,
        loss_func=nn.CrossEntropyLoss(),
    )

    trainer = Trainer(fixed_params=fixed_params)
    results = trainer.train(model_params=qnn_params)

    save_results(results, "fusion_5.json")
