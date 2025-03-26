import logging
from models.model import TwoLayerModel
from training.train import Train
from config import OptunaParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    trainer = Train()
    best_params = trainer.optimize_hyperparameters(n_trials=5)
    logger.info(f"Best hyperparameters: {best_params}")

    optuna_params = OptunaParams(**best_params)
    trainer.train(optuna_params)

    best_model = TwoLayerModel(
        fixed_params=trainer.fixed_params, optuna_params=optuna_params
    )
    trainer.log_mlflow_model(best_model)

if __name__ == "__main__":
    main()