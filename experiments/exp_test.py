import logging
from sklearn.datasets import load_diabetes
import torch
import torch.nn as nn

from config import FixedParams, NNParams, QNNParams, QSVMParams, SVMParams
from evaluation import compare, t_student
from models.model_NN import TwoLayerModel
from models.model_QNN import VariationalQuantumCircuit
from models.model_QSVM import QSVM
from models.model_SVM import SVM
from training.train import Trainer
from utils.data_loader import load_data, normalize_L2, scale_min_max, scale_to_pi
from utils.optuna_utils import (
    optimize_hyperparameters_QNN,
    optimize_hyperparameters_QSVM,
    optimize_hyperparameters_SVM,
)
from utils.parameters_loader import load_params, save_params, save_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_scaling() -> bool:
    fixed = FixedParams()
    X, _ = load_data(fixed.dataset_file)
    data = X[:10]
    print("Original data:")
    print(data)

    data_pi = scale_to_pi(data)
    print("\n\n SCALE TO PI:")
    print(data_pi)
    print(f"max_1: {data_pi[:, :10].max()}")
    assert torch.pi == data_pi[:, :10].max()
    print(f"min_1: {data_pi[:, :10].min()}")
    assert -torch.pi == data_pi[:, :10].min()
    print(f"max_2: {data_pi[:, 10:].max()}")
    assert torch.pi == data_pi[:, 10:].max()
    print(f"min_2: {data_pi[:, 10:].min()}")
    assert -torch.pi == data_pi[:, 10:].min()

    data_min_max = scale_min_max(data)
    print("\n\n SCALE TO MIN MAX:")
    print(data_min_max)
    print(f"max_1: {data_min_max[:, :10].max()}")
    assert 1 == data_min_max[:, :10].max()
    print(f"min_1: {data_min_max[:, :10].min()}")
    assert 0 == data_min_max[:, :10].min()
    print(f"max_2: {data_min_max[:, 10:].max()}")
    assert 1 == data_min_max[:, 10:].max()
    print(f"min_2: {data_min_max[:, 10:].min()}")
    assert 0 == data_min_max[:, 10:].min()

    data_L2 = normalize_L2(data)
    print("\n\n SCALE TO L2:")
    print(data_L2)
    for record in data_L2:
        print(record[:10])
        print(f"Norm_1: {record[:10].norm(p=2)}")
        assert 1 == record[:10].norm(p=2)
        print(record[10:])
        print(f"Norm_2: {record[10:].norm(p=2)}")
        assert 1 == record[10:].norm(p=2)

    return True
