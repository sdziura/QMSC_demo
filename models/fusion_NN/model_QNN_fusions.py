import pennylane as qml
from pennylane import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

from config import FixedParams, QNNParams
from utils import quantum_utils


class VQC_fusion_1(pl.LightningModule):
    def __init__(self, fixed_params: FixedParams, QNN_params: QNNParams):
        super(VQC_fusion_1, self).__init__()
        self.fixed_params = fixed_params
        self.model_params = QNN_params

        # Device selection
        self.dev = quantum_utils.quantum_device(model_params=QNN_params)

        # Classical layer
        # self.classical_layer = nn.Linear(fixed_params.input_size, QNN_params.n_qubits)
        # self.relu = nn.ReLU()

        # Quantum parameters
        if self.model_params.ansatz_version in {3, 4}:
            weight_shapes = (QNN_params.n_layers, QNN_params.n_qubits, 3)
        else:
            weight_shapes = (QNN_params.n_layers, QNN_params.n_qubits)
        self.q_params = nn.Parameter(torch.rand(weight_shapes) * np.pi)

        # Loss and metrics
        self.loss_fn = QNN_params.loss_func
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.F1 = torchmetrics.F1Score(task="binary")

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

    def ansatz_1(self, weights, axis):
        """
        Defines the quantum ansatz (circuit structure).
        """
        q = range(self.model_params.n_qubits)
        for i in q:
            self.apply_rotation(axis, weights[i], wires=i)
        for i in q:
            qml.CNOT(wires=[i, i + 1])

    def ansatz_2(self, weights, axis):
        """
        Defines the quantum ansatz (circuit structure).
        """
        q = range(self.model_params.n_qubits)
        for i in q[: len(q) // 2]:
            self.apply_rotation(axis, weights[i], wires=i)
        for i in q[: len(q) // 2 - 1]:
            qml.CNOT(wires=[i, i + 1])

        for i in q[len(q) // 2 :]:
            self.apply_rotation(axis, weights[i], wires=i)
        for i in q[len(q) // 2 : -1]:
            qml.CNOT(wires=[i, i + 1])

    def ansatz_3(self, weights):
        """
        Defines the quantum ansatz (circuit structure).
        """
        q = range(self.model_params.n_qubits)
        qml.StronglyEntanglingLayers(
            weights=weights[:, : len(q) // 2, :], wires=q[: len(q) // 2]
        )
        qml.StronglyEntanglingLayers(
            weights=weights[:, len(q) // 2 :, :], wires=q[len(q) // 2 :]
        )

    def ansatz_4(self, weights):
        """
        Defines the quantum ansatz (circuit structure).
        """
        q = range(self.model_params.n_qubits)
        qml.StronglyEntanglingLayers(weights=weights, wires=q)

    def embedding_1(self, x):
        qml.AngleEmbedding(
            x[: len(x) // 2],
            wires=range(self.model_params.n_qubits),
            rotation=self.model_params.embedding_axis,
        )
        qml.AngleEmbedding(
            x[len(x) // 2 :],
            wires=range(self.model_params.n_qubits),
            rotation=self.model_params.embedding_axis_2,
        )

    def embedding_2(self, x):
        qml.AmplitudeEmbedding(
            x,
            wires=range(self.model_params.n_qubits),
            pad_with=True,
        )

    def embedding_3(self, x):
        qml.AmplitudeEmbedding(
            x[: len(x) // 2],
            wires=range(self.model_params.n_qubits // 2),
            pad_with=True,
        )
        qml.AmplitudeEmbedding(
            x[len(x) // 2 :],
            wires=range(self.model_params.n_qubits // 2, self.model_params.n_qubits),
            pad_with=True,
        )

    def variational_circuit(self, weights, x=None):
        """
        Defines the quantum circuit with parameterized gates.
        """

        @qml.qnode(self.dev, interface="torch")
        def circuit(weights, x):
            if self.model_params.embedding_version == 1:
                self.embedding_1(x=x)
            elif self.model_params.embedding_version == 2:
                self.embedding_2(x=x)
            elif self.model_params.embedding_version == 3:
                self.embedding_3(x=x)

            # Set the axis of rotations for each layer of ansatz
            rot_axis = ["Y"] * self.model_params.n_layers
            rot_axis[0] = self.model_params.rot_axis_0
            if self.model_params.n_layers == 2:
                rot_axis[1] = self.model_params.rot_axis_1

            # If taken ansatz_version_2, the ansatz are seperated for half of qubits,
            # so the output is the average after those 2 ansatz
            if self.model_params.ansatz_version == 1:
                for i in range(self.model_params.n_layers):
                    self.ansatz_1(weights[i], rot_axis[i])
                return (
                    qml.expval(qml.PauliZ(wires=0)),
                    qml.expval(qml.PauliZ(wires=1)),
                )
            elif self.model_params.ansatz_version == 2:
                for i in range(self.model_params.n_layers):
                    self.ansatz_2(weights[i], rot_axis[i])
                return (
                    qml.expval(qml.PauliZ(wires=0)),
                    qml.expval(qml.PauliZ(wires=self.model_params.n_qubits // 2)),
                )
            elif self.model_params.ansatz_version == 3:
                self.ansatz_3(weights)
                return (
                    qml.expval(qml.PauliZ(wires=0)),
                    qml.expval(qml.PauliZ(wires=self.model_params.n_qubits // 2)),
                )
            elif self.model_params.ansatz_version == 4:
                self.ansatz_4(weights)
                return (
                    qml.expval(qml.PauliZ(wires=0)),
                    qml.expval(qml.PauliZ(wires=1)),
                )

        return circuit(weights, x)

    def forward(self, x):
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
        f1 = self.F1(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_f1", f1, prog_bar=True, on_epoch=True, on_step=False)
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

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
