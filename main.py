import logging
from evaluation.compare import compare
from experiments import exp_compare, exp_fusion
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
from config import FixedParams, NNParams, QNNParams, SVMParams, QSVMParams


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the training process.
    nohup python main.py > output.log 2>&1 &
    """
    # CUDA settings
    cuda_utils.set_cuda()
    exp_fusion.fusion_2()


if __name__ == "__main__":
    main()  # 12543
