from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np

BINARY_ANSATZES = ("ry_cz", "ry_rz_cz", "strongly_entangling")
FRACTIONAL_ANSATZES = ("ry", "ry_cz", "ry_rz_cz")


def binary_hwe_ry_cz_ring(params: np.ndarray, depth: int, n_wires: int) -> None:
    """Hardware-efficient: RY layers + CZ ring entanglement."""
    for d in range(depth):
        for i in range(n_wires):
            qml.RY(params[d, i], wires=i)
        for i in range(n_wires - 1):
            qml.CZ(wires=[i, i + 1])
        if n_wires > 2:
            qml.CZ(wires=[n_wires - 1, 0])


def binary_hwe_ry_rz_cz_ring(params: np.ndarray, depth: int, n_wires: int) -> None:
    """Hardware-efficient: RY/RZ rotation layers + CZ ring entanglement."""
    for d in range(depth):
        for i in range(n_wires):
            qml.RY(params[d, i, 0], wires=i)
            qml.RZ(params[d, i, 1], wires=i)
        for i in range(n_wires - 1):
            qml.CZ(wires=[i, i + 1])
        if n_wires > 2:
            qml.CZ(wires=[n_wires - 1, 0])


def binary_strongly_entangling(params: np.ndarray, depth: int, n_wires: int) -> None:
    """PennyLane StronglyEntanglingLayers template."""
    qml.StronglyEntanglingLayers(params, wires=range(n_wires))


def binary_ansatz_shape(ansatz: str, depth: int, n_wires: int) -> tuple[int, ...]:
    if ansatz == "ry_cz":
        return (depth, n_wires)
    if ansatz == "ry_rz_cz":
        return (depth, n_wires, 2)
    if ansatz == "strongly_entangling":
        return (depth, n_wires, 3)
    raise ValueError(
        f"Unsupported binary ansatz {ansatz!r}; expected one of {BINARY_ANSATZES}."
    )


def apply_binary_ansatz(
    ansatz: str, params: np.ndarray, depth: int, n_wires: int
) -> None:
    if ansatz == "ry_cz":
        binary_hwe_ry_cz_ring(params, depth=depth, n_wires=n_wires)
        return
    if ansatz == "ry_rz_cz":
        binary_hwe_ry_rz_cz_ring(params, depth=depth, n_wires=n_wires)
        return
    if ansatz == "strongly_entangling":
        binary_strongly_entangling(params, depth=depth, n_wires=n_wires)
        return
    raise ValueError(
        f"Unsupported binary ansatz {ansatz!r}; expected one of {BINARY_ANSATZES}."
    )


def fractional_ry_layer(thetas: np.ndarray, n_wires: int) -> None:
    """Baseline fractional ansatz: one or more RY layers per wire."""
    arr = np.array(thetas)
    if arr.ndim == 1:
        arr = arr.reshape(1, n_wires)
    for d in range(arr.shape[0]):
        for i in range(n_wires):
            qml.RY(arr[d, i], wires=i)


def fractional_ry_cz_ring(params: np.ndarray, depth: int, n_wires: int) -> None:
    """Fractional VQE ansatz: RY layers + CZ ring entanglement."""
    binary_hwe_ry_cz_ring(params, depth=depth, n_wires=n_wires)


def fractional_ry_rz_cz_ring(params: np.ndarray, depth: int, n_wires: int) -> None:
    """Fractional VQE ansatz: RY/RZ layers + CZ ring entanglement."""
    binary_hwe_ry_rz_cz_ring(params, depth=depth, n_wires=n_wires)


def fractional_ansatz_shape(ansatz: str, depth: int, n_wires: int) -> tuple[int, ...]:
    if ansatz == "ry":
        return (depth, n_wires)
    if ansatz == "ry_cz":
        return (depth, n_wires)
    if ansatz == "ry_rz_cz":
        return (depth, n_wires, 2)
    raise ValueError(
        f"Unsupported fractional ansatz {ansatz!r}; expected one of {FRACTIONAL_ANSATZES}."
    )


def apply_fractional_ansatz(
    ansatz: str, params: np.ndarray, depth: int, n_wires: int
) -> None:
    if ansatz == "ry":
        fractional_ry_layer(params, n_wires=n_wires)
        return
    if ansatz == "ry_cz":
        fractional_ry_cz_ring(params, depth=depth, n_wires=n_wires)
        return
    if ansatz == "ry_rz_cz":
        fractional_ry_rz_cz_ring(params, depth=depth, n_wires=n_wires)
        return
    raise ValueError(
        f"Unsupported fractional ansatz {ansatz!r}; expected one of {FRACTIONAL_ANSATZES}."
    )
