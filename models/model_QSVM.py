import logging
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
from sklearn.svm import SVC
import pennylane as qml

from config import QSVMParams, FixedParams, ModelParams
from utils import quantum_utils
from utils.cuda_utils import check_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
multiprocessing.set_start_method("fork", force=True)


class QSVM:
    def __init__(self, fixed_params: FixedParams, QSVM_params: QSVMParams):

        self.n_qubits = QSVM_params.n_qubits
        self.model_params = QSVM_params
        self.fixed_params = fixed_params
        # Device selection
        self.dev = quantum_utils.quantum_device(model_params=QSVM_params)

        # self.projector = np.zeros((2**QSVM_params.n_qubits, 2**QSVM_params.n_qubits))
        # self.projector[0, 0] = 1
        # self.projector = torch.tensor(self.projector)

        self.model = SVC(
            kernel=self.kernel_matrix,
            C=QSVM_params.C,
        )

    # def kernel(self, x1, x2):
    #     """The quantum kernel."""

    #     @qml.qnode(self.dev)
    #     def qkernel(x1, x2):
    #         self.feature_map(x1)
    #         qml.adjoint(self.feature_map)(x2)
    #         return qml.expval(qml.Hermitian(self.projector, wires=range(self.n_qubits)))

    #     return qkernel(x1, x2)

    # def kernel_matrix(self, A, B):
    #     def compute_row(a):
    #         return [self.kernel(a, b) for b in B]

    #     results = Parallel(n_jobs=-1)(
    #         delayed(compute_row)(a) for a in tqdm(A, desc="Computing Kernel Matrix")
    #     )
    #     return results

    ###########33
    # QNode that returns full statevector |phi(x)>

    def statevector(self, x1):
        """The quantum kernel."""

        @qml.qnode(self.dev, interface="autograd")
        def statevector_(x):
            if self.n_qubits == 10:
                self.feature_map(x)
            elif self.n_qubits == 20:
                qml.AngleEmbedding(x, range(self.n_qubits))
            else:
                raise ValueError(
                    f"Number of qubits is {self.n_qubits}. Should be 10 or 20"
                )

            return qml.state()

        return statevector_(x1)

    def compute_states(self, X):
        """Compute statevectors for all data points X."""
        states = []
        for x in tqdm(X):
            state = self.statevector(x)
            states.append(state)
        logger.info("State vectors ready")
        return np.array(
            states, dtype=np.complex128
        )  # Ensure correct dtype for complex numbers

    def kernel_matrix(self, X, Y=None):
        """Compute kernel matrix by inner products of statevectors."""
        states_X = self.compute_states(X)
        logger.info("State vectors returned as np arrays")
        if Y is None:
            logger.info("No Y data to compute states")
            states_Y = states_X
        else:
            logger.info("Computing state vectors for Y...")
            states_Y = self.compute_states(Y)

        # Kernel = |<phi(x)|phi(y)>|^2 = abs(dot product)^2
        logger.info("Calculating dot product...")
        kernel_mat = np.abs(np.dot(states_X, states_Y.conj().T)) ** 2
        logger.info("... Done!")
        return kernel_mat

    ################

    def feature_map(self, x):
        for i in range(self.n_qubits):
            qml.RX(x[i], wires=i)
            qml.RY(x[self.n_qubits + i], wires=i)

    def fit(self, X, y):
        self.model.fit(X=X, y=y)

    def predict(self, X):
        return self.model.predict(X=X)
