import pennylane as qml
from pennylane import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

from config import FixedParams, QNNParams


class VariationalQuantumCircuit(pl.LightningModule):
    def __init__(self, fixed_params: FixedParams, QNN_params: QNNParams):
        super(VariationalQuantumCircuit, self).__init__()
        self.fixed_params = fixed_params
        self.model_params = QNN_params

        # Device selection
        if fixed_params.use_gpu:
            self.dev = qml.device(
                "lightning.gpu", wires=QNN_params.n_qubits, shots=QNN_params.shots
            )
            print("Using GPU with lightning.gpu")
        else:
            self.dev = qml.device(
                "default.qubit", wires=QNN_params.n_qubits, shots=QNN_params.shots
            )
            print("Using CPU with default.qubit")

        # Classical layer
        self.classical_layer = nn.Linear(fixed_params.input_size, QNN_params.n_qubits)
        self.relu = nn.ReLU()

        # Quantum parameters
        weight_shapes = (QNN_params.n_layers, QNN_params.n_qubits)
        self.q_params = nn.Parameter(torch.rand(weight_shapes) * np.pi)

        # Loss and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=fixed_params.output_size
        )
        self.lr = QNN_params.learning_rate  # Learning rate from QNN_params

    def apply_rotation(self, axis: str, angle: float, wires: int):
        if axis == "X":
            qml.RX(angle, wires=wires)
        elif axis == "Y":
            qml.RY(angle, wires=wires)
        elif axis == "Z":
            qml.RZ(angle, wires=wires)
        else:
            raise ValueError(f"Invalid axis '{axis}'. Must be 'X', 'Y', or 'Z'.")

    def ansatz(self, weights, axis):
        """
        Defines the quantum ansatz (circuit structure).
        """
        for i in range(self.model_params.n_qubits):
            self.apply_rotation(axis, weights[i], wires=i)
        for i in range(self.model_params.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    def variational_circuit(self, weights, x=None):
        """
        Defines the quantum circuit with parameterized gates.
        """

        @qml.qnode(self.dev, interface="torch")
        def circuit(weights, x):
            qml.AngleEmbedding(
                x,
                wires=range(self.model_params.n_qubits),
                rotation=self.model_params.embedding_axis,
            )

            rot_axis = ["Y"] * self.model_params.n_layers
            rot_axis[0] = self.model_params.rot_axis_0
            if self.model_params.n_layers == 2:
                rot_axis[1] = self.model_params.rot_axis_1

            for i in range(self.model_params.n_layers):
                self.ansatz(weights[i], rot_axis[i])
            return (
                qml.expval(qml.PauliZ(wires=0)),
                qml.expval(qml.PauliZ(wires=1)),
            )

        return circuit(weights, x)

    def forward(self, x):
        # Pass input through the classical layer
        x = self.classical_layer(x)
        x = self.relu(x)

        # Pass the output of the classical layer to the quantum circuit
        outputs = []
        for input_sample in x:
            output = self.variational_circuit(self.q_params, input_sample)
            output = torch.stack(output)
            outputs.append(output)
        return torch.stack(outputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        self.log("test_acc", acc, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
