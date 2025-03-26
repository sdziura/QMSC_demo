from dataclasses import dataclass

@dataclass
class FixedParams:
    folds: int = 2
    random_state: int = 42
    batch_size: int = 32
    val_check_interval: int = 1
    log_every_n_steps: int = 1
    input_size: int = 20
    output_size: int = 2

@dataclass
class OptunaParams:
    max_iter: int = 1
    learning_rate: float = 0.001
    hidden_size_1: int = 32
    hidden_size_2: int = 64
    hidden_size_3: int = 32