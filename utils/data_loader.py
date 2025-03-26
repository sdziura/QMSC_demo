import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset


def load_data(file_path: str):
    with h5py.File(file_path, "r") as f:
        labels, observations = f["label"][:], f["observed"][:]

    observations = observations.reshape(observations.shape[0], -1)
    X = torch.tensor(observations, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    return X, y


def get_dataloader(
    dataset: TensorDataset, indices: np.ndarray, shuffle: bool, batch_size: int
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
    )
