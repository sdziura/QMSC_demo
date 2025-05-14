import logging
from evaluation.compare import compare
from models.model_NN import TwoLayerModel
from models.model_QNN import VariationalQuantumCircuit
from training.train import Trainer
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
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision("medium")
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    # trainer = Trainer()

    # Optimize hyperparameters
    # best_params = optimize_hyperparameters_QNN(
    #     trainer, n_trials=trainer.fixed_params.optuna_trials
    # )
    # logger.info(f"Best hyperparameters: {best_params}")
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

    # save_params(best_params, "QNN_best_params")
    # Train the model with the best hyperparameters
    # optuna_params = OptunaParams(**best_params)
    # trainer.train(optuna_params)


if __name__ == "__main__":
    main()
