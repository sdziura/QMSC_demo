import torch
import mlflow
from mlflow.models import infer_signature
import mlflow.pytorch


def initialize_mlflow(tracking_uri: str, experiment_name: str):
    """
    Initialize MLflow with the given tracking URI and experiment name.

    Parameters
    ----------
    tracking_uri : str
        The URI for the MLflow tracking server.
    experiment_name : str
        The name of the experiment to log runs under.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()


def log_mlflow_model(
    model: torch.nn.Module, input_size: int, model_name: str = "best_model"
) -> None:
    """
    Logs the trained model to MLFlow.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model to be logged.
    input_size : int
        The size of the input tensor.
    model_name : str, optional
        The name of the model to be logged, by default "best_model".
    """
    input_example = torch.randn(1, input_size)
    input_example_np = input_example.numpy()
    signature = infer_signature(input_example_np, model(input_example).detach().numpy())

    mlflow.pytorch.log_model(
        model,
        model_name,
        input_example=input_example_np,
        signature=signature,
    )


def log_mlflow_params(params: dict) -> None:
    """
    Logs parameters to MLFlow.

    Parameters
    ----------
    params : dict
        A dictionary of parameters to be logged.
    """
    for key, value in params.items():
        mlflow.log_param(key, value)
