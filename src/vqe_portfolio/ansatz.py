from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np


def binary_hwe_ry_cz_ring(params: np.ndarray, depth: int, n_wires: int) -> None:
    """Hardware-efficient: RY layers + CZ ring entanglement."""
    for d in range(depth):
        for i in range(n_wires):
            qml.RY(params[d, i], wires=i)
        for i in range(n_wires - 1):
            qml.CZ(wires=[i, i + 1])
        if n_wires > 2:
            qml.CZ(wires=[n_wires - 1, 0])


def fractional_ry_layer(thetas: np.ndarray, n_wires: int) -> None:
    """Baseline fractional ansatz: single RY per wire."""
    for i in range(n_wires):
        qml.RY(thetas[i], wires=i)
