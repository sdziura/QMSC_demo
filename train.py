import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold

from model import TwoLayerModel


def load_data(file_path="hmm_gaussian_chains.h5"):
    with h5py.File(file_path, "r") as f:
        y = f["label"][:]
        observations = f["observed"][:]

    # Flatten each (10,2) sequence into a 1D vector of size 20
    observations = observations.reshape(observations.shape[0], -1)
    X = torch.tensor(observations, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    print("Observations shape:", observations.shape)
    print("Labels shape:", y.shape)
    return X, y


def train(X, y):
    # Convert dataset into a list of indices for cross-validation
    dataset = TensorDataset(X, y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create a PyTorch Dataset & DataLoader
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=True)

        model = TwoLayerModel(input_size=20)

        trainer = pl.Trainer(max_epochs=10, accelerator="auto", val_check_interval=1)
        trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    X, y = load_data()
    train(X, y)