from torch import nn
import pytorch_lightning as pl
import torch.optim as optim

from params import FixedParams, OptunaParams


class TwoLayerModel(pl.LightningModule):
    """
    A PyTorch Lightning Module implementing a neural network with two hidden layers.

    Args:
        fixed_params (FixedParams): An object containing fixed parameters such as input and output sizes.
        optuna_params (OptunaParams): An object containing hyperparameters to be optimized, such as learning rate and hidden layer sizes.

    Attributes:
        lr (float): Learning rate for the optimizer.
        model (nn.Sequential): The neural network model consisting of linear layers and ReLU activations.
        loss_fn (nn.CrossEntropyLoss): The loss function used for training and evaluation.

    Methods:
        forward(x):
            Performs a forward pass through the network.

        training_step(batch):
            Defines the training step, including the forward pass and loss computation.

        test_step(batch):
            Defines the test step, including the forward pass and loss computation.

        validation_step(batch):
            Defines the validation step, including the forward pass and loss computation.

        configure_optimizers():
            Configures the optimizer for training.
    """

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
