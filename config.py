from dataclasses import dataclass, field
from torch import nn


@dataclass
class FixedParams:
    """
    A dataclass containing fixed hyperparameters for training.

    Attributes
    ----------
    folds : int
        Number of folds for cross-validation.
    random_state : int
        Random state for reproducibility.
    val_check_interval : int
        Interval for validation checks.
    input_size : int
        Size of the input features.
    output_size : int
        Size of the output features.
    max_epochs : int
        Maximum number of epochs for training.
    experiment_name : str
        Name of the MLFlow experiment.
    dataset_file : str
        Path to the dataset file.
    optuna_trials : int
        Number of Optuna trials for hyperparameter optimization.
    """

    folds: int = 5
    random_state: int = 42
    val_check_interval: float | int = 1.0
    input_size: int = 20
    output_size: int = 2
    max_epochs: int = 200
    experiment_name: str = "HMM_Classification_QSVM"
    dataset_file: str = "data/hmm_gaussian_chains.h5"
    mlflow_uri: str = "http://127.0.0.1:8080"
    optuna_trials: int = 10
    use_gpu = False
    profiler_active_steps = 0


@dataclass
class ModelParams:
    pass  # acts as a base type for polymorphism


@dataclass
class NNParams(ModelParams):
    """
    A dataclass containing hyperparameters to be optimized by Optuna.

    Attributes
    ----------
    learning_rate : float
        Learning rate for the optimizer.
    hidden_size_1 : int
        Size of the first hidden layer.
    hidden_size_2 : int
        Size of the second hidden layer.
    hidden_size_3 : int
        Size of the third hidden layer.
    batch_size : int
        Batch size for training.
    dropout : float
        Dropout rate for regularization.
    """

    model_type: str = "nn"
    model_name: str = "TwoLayerModel"
    learning_rate: float = 0.0005
    hidden_size_1: int = 32
    hidden_size_2: int = 64
    batch_size: int = 32
    dropout: float = 0.2
    loss_func: nn.Module = field(default_factory=nn.CrossEntropyLoss)


@dataclass
class SVMParams(ModelParams):
    model_type: str = "svm"
    model_name: str = "SVM"
    C: float = 1.0
    kernel: str = "rbf"
    gamma: str = "scale"
    degree: int = 3


@dataclass
class QNNParams(ModelParams):
    model_type: str = "qnn"
    model_name: str = "VariationalQuantumCircuit"
    learning_rate: float = 0.001
    batch_size: int = 256
    n_layers: int = 2  # max 2 for now
    n_qubits: int = 10
    ansatz_version: int = 1
    embedding_version: int = 1
    embedding_axis: str = "X"
    embedding_axis_2: str = "Y"
    rot_axis_0: str = "X"
    rot_axis_1: str = "Y"
    shots: int = None
    loss_func: nn.Module = field(default_factory=nn.CrossEntropyLoss)


@dataclass
class QSVMParams(ModelParams):
    model_type: str = "qsvm"
    model_name: str = "QSVM"
    n_qubits: int = 10
    embedding_type: int = 1
    embedding_axis: str = "X"
    C: float = 1.0
    shots: int = None
