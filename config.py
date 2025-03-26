from dataclasses import dataclass


@dataclass
class FixedParams:
    folds: int = 5
    random_state: int = 42
    val_check_interval: int = 1
    log_every_n_steps: int = 1
    input_size: int = 20
    output_size: int = 2
    max_epochs: int = 500
    experiment_name: str = "HMM_Classification_todelete"
    dataset_file: str = "data/hmm_gaussian_chains.h5"
    optuna_trials: int = 50


@dataclass
class OptunaParams:
    learning_rate: float = 0.001
    hidden_size_1: int = 32
    hidden_size_2: int = 64
    hidden_size_3: int = 32
    batch_size: int = 32
    dropout: float = 0.5
