import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
import pennylane as qml

from config import QSVMParams, FixedParams, ModelParams
from utils import quantum_utils


class QSVM:
    def __init__(self, fixed_params: FixedParams, QSVM_params: QSVMParams):

        self.n_qubits = QSVM_params.n_qubits
        self.model_params = QSVM_params
        self.fixed_params = fixed_params
        # Device selection
        self.dev = quantum_utils.quantum_device(model_params=QSVM_params)

        self.projector = np.zeros((2**QSVM_params.n_qubits, 2**QSVM_params.n_qubits))
        self.projector[0, 0] = 1

        self.model = SVC(
            kernel=self.kernel_matrix,
            C=QSVM_params.C,
        )

    def kernel(self, x1, x2):
        """The quantum kernel."""

        @qml.qnode(self.dev)
        def qkernel(x1, x2):
            self.feature_map(x1)
            qml.adjoint(self.feature_map)(x2)
            return qml.expval(qml.Hermitian(self.projector, wires=range(self.n_qubits)))

        return qkernel(x1, x2)

    def kernel_matrix(self, A, B):
        """Compute the matrix whose entries are the kernel
        evaluated on pairwise data from sets A and B."""
        if FixedParams.use_gpu:
            A = A.to("cuda")
            B = B.to("cuda")
        return np.array([[self.kernel(a, b) for b in B] for a in tqdm(A)])

    def feature_map(self, x):
        for i in range(self.n_qubits):
            qml.RX(x[i], wires=i)
            qml.RY(x[self.n_qubits + i], wires=i)

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predict(self, X):
        return self.model.predict(X=X)
