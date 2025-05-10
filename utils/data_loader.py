import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset


def load_data(file_path: str):
    with h5py.File(file_path, "r") as f:
        labels, observations = f["label"][:], f["observed"][:]

    observations = observations.reshape(observations.shape[0], -1)
    X = torch.tensor(observations, dtype=torch.float32).to("cuda")
    y = torch.tensor(labels, dtype=torch.long).to("cuda")

    return X, y


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
        num_workers=4,
    )
