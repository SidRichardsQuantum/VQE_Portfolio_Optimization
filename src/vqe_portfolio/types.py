from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class BinaryVQEConfig:
    depth: int = 2
    steps: int = 75
    stepsize: float = 0.3
    log_every: int = 5

    # Constraint / objective params
    lam: float = 5.0
    alpha: float = 2.0
    k: int = 2

    # Quantum execution
    device: str = "default.qubit"
    shots_train: Optional[int] = None      # None = exact expectations
    shots_sample: int = 2000

    # Reproducibility
    seed: int = 0


@dataclass(frozen=True)
class FractionalVQEConfig:
    steps: int = 75
    stepsize: float = 0.3
    log_every: int = 5

    lam: float = 5.0

    device: str = "default.qubit"
    shots: Optional[int] = None

    seed: int = 0


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


@dataclass(frozen=True)
class FractionalVQEResult:
    thetas: np.ndarray
    cost_trace: OptimizeTrace
    weights: np.ndarray

    lambdas: np.ndarray | None = None
    allocs_by_lambda: np.ndarray | None = None  # shape (L, n)
