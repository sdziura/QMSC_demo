import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl

from model import TwoLayerModel


# Load data
with h5py.File("hmm_gaussian_chains.h5", "r") as f:
    y = f["label"][:]
    observations = f["observed"][:]

# Flatten each (10,2) sequence into a 1D vector of size 20
observations = observations.reshape(observations.shape[0], -1)
observations = torch.tensor(observations, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print("Observations shape:", observations.shape)
print("Labels shape:", y.shape)


# Create a PyTorch Dataset & DataLoader
dataset = TensorDataset(observations, y)
train, val, test = random_split(
    dataset, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader = DataLoader(test, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=True)

model = TwoLayerModel(input_size=20)

trainer = pl.Trainer(max_epochs=10, accelerator="auto", val_check_interval=1)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
