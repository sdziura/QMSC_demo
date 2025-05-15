import logging
from evaluation.compare import compare
from experiments import exp_compare
from models.model_NN import TwoLayerModel
from models.model_QNN import VariationalQuantumCircuit
from training.train import Trainer
from utils import cuda_utils
from utils.optuna_utils import (
    optimize_hyperparameters_NN,
    optimize_hyperparameters_SVM,
    optimize_hyperparameters_QNN,
)
from utils.parameters_loader import save_params, load_params, save_results
import torch
from config import FixedParams, NNParams, QNNParams


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the training process.
    """
    # CUDA settings
    cuda_utils.set_cuda()
    exp_compare.find_best_qsvm()


if __name__ == "__main__":
    main()
