from torch import nn
import pytorch_lightning as pl
import torch.optim as optim

from train import FixedParams, OptunaParams


class TwoLayerModel(pl.LightningModule):
    def __init__(self, fixed_params: FixedParams, optuna_params: OptunaParams):
        super(TwoLayerModel, self).__init__()
        self.lr = optuna_params.learning_rate
        self.model = nn.Sequential(
            nn.Linear(fixed_params.input_size, optuna_params.hidden_size_1),
            nn.ReLU(),
            nn.Linear(optuna_params.hidden_size_1, optuna_params.hidden_size_2),
            nn.ReLU(),
            nn.Linear(optuna_params.hidden_size_2, optuna_params.hidden_size_3),
            nn.ReLU(),
            nn.Linear(optuna_params.hidden_size_3, fixed_params.output_size),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
