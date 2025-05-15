import numpy as np
import pennylane as qml

from config import FixedParams, ModelParams


def scale_to_pi(x):
    x = np.array(x)
    x_min, x_max = np.min(x), np.max(x)
    return (x - x_min) / (x_max - x_min) * 2 * np.pi - np.pi


def quantum_device(
    model_params: ModelParams, fixed_params: FixedParams = FixedParams()
):
    if fixed_params.use_gpu:
        dev = qml.device(
            "lightning.gpu", wires=model_params.n_qubits, shots=model_params.shots
        )
        print("Using GPU with lightning.gpu")
    else:
        dev = qml.device(
            "default.qubit", wires=model_params.n_qubits, shots=model_params.shots
        )
        print("Using CPU with default.qubit")

    return dev
