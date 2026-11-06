from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class BinaryVQEConfig:
    depth: int = 2
    ansatz: str = "ry_cz"
    steps: int = 75
    stepsize: float = 0.3
    log_every: int = 5

    # Constraint / objective params
    lam: float = 5.0
    alpha: float | str = 2.0
    k: int = 2

    # Quantum execution
    device: str = "default.qubit"
    shots_train: Optional[int] = None  # None = exact expectations
    shots_sample: int = 2000

    # Reproducibility
    seed: int = 0

    # Opt-in reliability controls; legacy defaults remain reproducible.
    restarts: int = 1
    normalize_energy: bool = False
    keep_best: bool = False
    stepsize_decay: float = 0.0


@dataclass(frozen=True)
class QAOAConfig:
    p: int = 2
    steps: int = 75
    stepsize: float = 0.3
    log_every: int = 5

    # Constraint / objective params
    lam: float = 5.0
    alpha: float | str = 2.0
    k: int = 2

    # Mixer: "x" for standard QAOA, "xy" for fixed-cardinality style mixing
    mixer: str = "x"

    # Quantum execution
    device: str = "default.qubit"
    shots_train: Optional[int] = None  # None = exact expectations
    shots_sample: int = 2000

    # Reproducibility
    seed: int = 0
    xy_initial_state: str = "basis"  # "dicke": uniform exact-K superposition
    xy_topology: str = "ring"  # or "complete"


@dataclass(frozen=True)
class VariationalAllocatorConfig:
    depth: int = 1
    ansatz: str = "ry"
    steps: int = 75
    stepsize: float = 0.3
    log_every: int = 5

    lam: float = 5.0

    device: str = "default.qubit"
    shots: Optional[int] = None

    seed: int = 0


# Compatibility alias for the historical allocator name.
FractionalVQEConfig = VariationalAllocatorConfig


@dataclass(frozen=True)
class LambdaSweepConfig:
    lambdas: Sequence[float]
    steps_per_lambda: int = 80
    stepsize: float = 0.2
    warm_start: bool = False


@dataclass(frozen=True)
class OptimizeTrace:
    steps: Sequence[int]
    values: Sequence[float]


@dataclass(frozen=True)
class BinaryVQEResult:
    params: np.ndarray
    energy_trace: OptimizeTrace
    z_expect: np.ndarray
    x_prob: np.ndarray
    x_round: np.ndarray
    x_topk: np.ndarray

    # Sampling outputs
    sample_counts: dict[str, int]
    x_mode: np.ndarray
    x_best_feasible: np.ndarray | None

    # Lambda sweep
    lambdas: np.ndarray | None = None
    probs_by_lambda: np.ndarray | None = None  # shape (L, n)
    effective_alpha: float | None = None
    energy_scale: float = 1.0
    restart_energies: Sequence[float] = ()


@dataclass(frozen=True)
class QAOAResult:
    params: np.ndarray
    gammas: np.ndarray
    betas: np.ndarray
    cost_trace: OptimizeTrace
    state_probs: np.ndarray
    x_prob: np.ndarray
    x_round: np.ndarray
    x_topk: np.ndarray

    # Sampling outputs
    sample_counts: dict[str, int]
    x_mode: np.ndarray
    x_best_feasible: np.ndarray | None

    # Lambda sweep
    lambdas: np.ndarray | None = None
    probs_by_lambda: np.ndarray | None = None  # shape (L, n)


@dataclass(frozen=True)
class VariationalAllocatorResult:
    thetas: np.ndarray
    cost_trace: OptimizeTrace
    weights: np.ndarray

    lambdas: np.ndarray | None = None
    allocs_by_lambda: np.ndarray | None = None  # shape (L, n)


FractionalVQEResult = VariationalAllocatorResult
