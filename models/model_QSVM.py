import numpy as np
import torch
from torch.nn.functional import relu

from sklearn.svm import SVC

import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from config import QSVMParams, FixedParams, ModelParams


class QSVM:
    def __init__(self, fixed_params: FixedParams, QSVM_params: QSVMParams):

        self.n_qubits = QSVM_params.n_qubits
        self.model_params = QSVM_params
        self.fixed_params = fixed_params
        # Device selection
        if fixed_params.use_gpu:
            self.dev = qml.device(
                "lightning.gpu", wires=QSVM_params.n_qubits, shots=QSVM_params.shots
            )
            print("Using GPU with lightning.gpu")
        else:
            self.dev = qml.device(
                "default.qubit", wires=QSVM_params.n_qubits, shots=QSVM_params.shots
            )
            print("Using CPU with default.qubit")

        self.projector = np.zeros((2**QSVM_params.n_qubits, 2**QSVM_params.n_qubits))
        self.projector[0, 0] = 1

        self.model = SVC(
            kernel=self.kernel,
            C=QSVM_params.C,
        )

    def kernel(self, x1, x2):
        """The quantum kernel."""

        @qml.qnode(self.dev)
        def qkernel(x1, x2):
            AngleEmbedding(x1, wires=range(self.n_qubits))
            qml.adjoint(AngleEmbedding)(x2, wires=range(self.n_qubits))
            return qml.expval(qml.Hermitian(self.projector, wires=range(self.n_qubits)))

        return qkernel(x1, x2)

    def kernel_matrix(self, A, B):
        """Compute the matrix whose entries are the kernel
        evaluated on pairwise data from sets A and B."""
        return np.array([[self.kernel(a, b) for b in B] for a in A])

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predict(self, X):
        return self.model.predict(X=X)
