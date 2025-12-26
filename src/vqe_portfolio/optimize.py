from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pennylane as qml
from pennylane import numpy as np


@dataclass(frozen=True)
class OptimizeResult:
    params: np.ndarray
    steps: list[int]
    values: list[float]


def adam_optimize(
    cost_fn: Callable[[np.ndarray], np.ndarray],
    init_params: np.ndarray,
    steps: int,
    stepsize: float,
    log_every: int = 5,
    callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
) -> OptimizeResult:
    """
    Generic Adam loop using PennyLane's AdamOptimizer.
    Records cost every log_every steps.
    """
    opt = qml.AdamOptimizer(stepsize=stepsize)

    params = init_params
    hist_steps: list[int] = []
    hist_vals: list[float] = []

    for t in range(steps):
        params, val = opt.step_and_cost(cost_fn, params)
        if (t + 1) % log_every == 0:
            v = float(val)
            hist_steps.append(t + 1)
            hist_vals.append(v)
            if callback is not None:
                callback(t + 1, v, params)

    return OptimizeResult(params=params, steps=hist_steps, values=hist_vals)
