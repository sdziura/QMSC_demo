import logging
from training.train import Trainer
from utils.optuna_utils import optimize_hyperparameters_NN, optimize_hyperparameters_SVM
from utils.parameters_loader import save_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the training process.
    """
    trainer = Trainer()

    # Optimize hyperparameters
    best_params = optimize_hyperparameters_SVM(
        trainer, n_trials=trainer.fixed_params.optuna_trials
    )
    logger.info(f"Best hyperparameters: {best_params}")

    save_params(best_params, "SVM_best_params")
    # Train the model with the best hyperparameters
    # optuna_params = OptunaParams(**best_params)
    # trainer.train(optuna_params)


if __name__ == "__main__":
    main()
