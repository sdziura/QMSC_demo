from torch import nn
import pytorch_lightning as pl
import torch.optim as optim

class Two_Layer_Model(pl.LightningModule):
    def __init__(self, input_size=20, lr=0.001):
        super(Two_Layer_Model, self).__init__()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Rozmiar wyjściowy to 2, ponieważ mamy 2 klasy 
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

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)