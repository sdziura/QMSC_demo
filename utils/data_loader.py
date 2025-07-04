import logging
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

from config import FixedParams, ModelParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str):
    with h5py.File(file_path, "r") as f:
        labels, observations = f["label"][:], f["observed"][:]

    observations = observations.reshape(observations.shape[0], -1)
    X = torch.tensor(observations, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    return X, y


def normalize_data(x: torch.Tensor, model_params: ModelParams) -> torch.Tensor:
    if model_params.model_type in {"qnn", "qsvm"}:
        if model_params.embedding_version == 1:
            return scale_to_pi(X=x)
        else:
            return x
    else:
        return scale_min_max(X=x)


def scale_to_pi(X: torch.Tensor) -> torch.Tensor:
    # Take values from each sensor separately
    sensor_1 = X[:, :10]
    sensor_2 = X[:, 10:]
    max_1, min_1 = sensor_1.max(), sensor_1.min()
    max_2, min_2 = sensor_2.max(), sensor_2.min()
    diff_1 = max_1 - min_1
    diff_2 = max_2 - min_2
    # Avoid dividing by 0
    if diff_1 == 0:
        sensor_1_corrected = torch.zeros_like(sensor_1)
    else:
        sensor_1_corrected = ((sensor_1 - min_1) * 2 * torch.pi / diff_1) - torch.pi

    if diff_2 == 0:
        sensor_2_corrected = torch.zeros_like(sensor_2)
    else:
        sensor_2_corrected = ((sensor_2 - min_2) * 2 * torch.pi / diff_2) - torch.pi
    logger.info("Data scaled to [-pi, pi]")
    return torch.cat([sensor_1_corrected, sensor_2_corrected], dim=1)


def scale_min_max(X: torch.Tensor) -> torch.Tensor:
    # Take values from each sensor separately
    sensor_1 = X[:, :10]
    sensor_2 = X[:, 10:]
    max_1, min_1 = sensor_1.max(), sensor_1.min()
    max_2, min_2 = sensor_2.max(), sensor_2.min()
    diff_1 = max_1 - min_1
    diff_2 = max_2 - min_2
    # Avoid dividing by 0
    if diff_1 == 0:
        sensor_1_corrected = torch.zeros_like(sensor_1)
    else:
        sensor_1_corrected = (sensor_1 - min_1) / diff_1

    if diff_2 == 0:
        sensor_2_corrected = torch.zeros_like(sensor_2)
    else:
        sensor_2_corrected = (sensor_2 - min_2) / diff_2
    logger.info("Data scaled to [0, 1]")
    return torch.cat([sensor_1_corrected, sensor_2_corrected], dim=1)


def normalize_L2(X: torch.Tensor) -> torch.Tensor:
    corrected_list = []
    for rec in X:
        sensor_1 = rec[:10]
        sensor_2 = rec[10:]
        norm_1 = torch.norm(sensor_1, p=2)
        if norm_1 == 0:
            sensor_1_corrected = torch.zeros_like(sensor_1)
        else:
            sensor_1_corrected = sensor_1 / norm_1
        norm_2 = torch.norm(sensor_2, p=2)
        if norm_2 == 0:
            sensor_2_corrected = torch.zeros_like(sensor_2)
        else:
            sensor_2_corrected = sensor_2 / norm_2
        rec_corrected = torch.cat([sensor_1_corrected, sensor_2_corrected], dim=0)
        corrected_list.append(rec_corrected)
    logger.info("Data normalized with L2 norm")
    return torch.stack(corrected_list)


def get_dataloader(
    dataset: TensorDataset, indices: np.ndarray, shuffle: bool, batch_size: int
) -> DataLoader:
    """
    Creates a DataLoader for a given dataset and indices.

    Parameters
    ----------
    dataset : TensorDataset
        The dataset to load.
    indices : np.ndarray
        The indices of the samples to load.
    shuffle : bool
        Whether to shuffle the data.
    batch_size : int
        The batch size for loading data.

    Returns
    -------
    DataLoader
        The DataLoader for the dataset.
    """
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
