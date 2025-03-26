import logging
from models.model import TwoLayerModel
from training.train import Train
from config import OptunaParams
from utils.mlflow_utils import log_mlflow_model, log_mlflow_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the training process.
    """
    trainer = Train()
    best_params = trainer.optimize_hyperparameters()
    logger.info(f"Best hyperparameters: {best_params}")

    optuna_params = OptunaParams(**best_params)
    trainer.train(optuna_params)

    best_model = TwoLayerModel(
        fixed_params=trainer.fixed_params, optuna_params=optuna_params
    )
    log_mlflow_params(best_params)
    log_mlflow_model(best_model, input_size=trainer.fixed_params.input_size)


if __name__ == "__main__":
    main()
