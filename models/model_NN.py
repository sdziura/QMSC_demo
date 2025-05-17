from torch import nn
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchmetrics

from config import FixedParams, ModelParams, NNParams


class TwoLayerModel(pl.LightningModule):
    """
        A PyTorch Lightning Module implementing a neural network with two hidden layers and dropout.

        Args
        ----
            fixed_params : FixedParams
    An object containing fixed parameters such as input and output sizes.
            optuna_params : OptunaParams
    An object containing hyperparameters to be optimized, such as learning rate, hidden layer sizes, and dropout.

        Attributes
        ----------
            lr : float
    Learning rate for the optimizer.
            model : nn.Sequential
    The neural network model consisting of linear layers, ReLU activations, and dropout.
            loss_fn : nn.CrossEntropyLoss
    The loss function used for training and evaluation.
            accuracy : torchmetrics.Accuracy
    The accuracy metric used for evaluation.

        Methods
        -------
            forward(x):
                Performs a forward pass through the network.
            training_step(batch, batch_idx):
                Defines the training step, including the forward pass and loss computation.
            test_step(batch, batch_idx):
                Defines the test step, including the forward pass and loss computation.
            validation_step(batch, batch_idx):
                Defines the validation step, including the forward pass and loss computation.
            configure_optimizers():
                Configures the optimizer for training.
    """

    def __init__(self, fixed_params: FixedParams, NN_params: NNParams):
        super(TwoLayerModel, self).__init__()
        self.model_params = NN_params
        self.fixed_params = fixed_params
        self.lr = NN_params.learning_rate
        self.model = nn.Sequential(
            nn.Linear(fixed_params.input_size, NN_params.hidden_size_1),
            nn.ReLU(),
            nn.Dropout(NN_params.dropout),
            nn.Linear(NN_params.hidden_size_1, NN_params.hidden_size_2),
            nn.ReLU(),
            nn.Dropout(NN_params.dropout),
            nn.Linear(NN_params.hidden_size_2, fixed_params.output_size),
        )
        self.loss_fn = NN_params.loss_func
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.F1 = torchmetrics.F1Score(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.F1(preds, y)
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        self.log("test_acc", acc, on_epoch=True, on_step=False)
        self.log("test_f1", f1, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.F1(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_f1", f1, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
