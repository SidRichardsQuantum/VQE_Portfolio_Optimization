from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as onp
import pennylane as qml
from pennylane import numpy as np


def report_phase(callback, phase):
    """Notify an optional execution timer without changing optimizer state."""
    if callback is not None:
        callback(phase)


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
    *,
    keep_best: bool = False,
    stepsize_decay: float = 0.0,
) -> OptimizeResult:
    """
    Generic Adam loop using PennyLane's AdamOptimizer.
    Records post-update costs and matching parameter snapshots every log_every
    steps and at the final step. keep_best selects by observed training cost,
    never by reference quality. Decay uses stepsize/(1 + decay*t).
    """
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if type(log_every) is not int or log_every < 1:
        raise ValueError("log_every must be a positive integer")
    if not np.isfinite(stepsize) or stepsize <= 0:
        raise ValueError("stepsize must be finite and positive")
    if not np.isfinite(stepsize_decay) or stepsize_decay < 0:
        raise ValueError("stepsize_decay must be finite and nonnegative")
    opt = qml.AdamOptimizer(stepsize=stepsize)
    best_params, best_value = init_params.copy(), float("inf")
    if keep_best:
        best_value = float(cost_fn(init_params))

    params = init_params
    hist_steps: list[int] = []
    hist_vals: list[float] = []

    for t in range(steps):
        opt.stepsize = stepsize / (1 + stepsize_decay * t)
        params, _ = opt.step_and_cost(cost_fn, params)
        record = (t + 1) % log_every == 0 or t + 1 == steps
        if keep_best or record:
            v = float(cost_fn(params))
            if not np.isfinite(v):
                raise ValueError("Training cost is nonfinite")
            if keep_best and v < best_value:
                best_params, best_value = params.copy(), v
        if record:
            hist_steps.append(t + 1)
            hist_vals.append(v)
            if callback is not None:
                callback(t + 1, v, params)

    return OptimizeResult(
        params=best_params if keep_best else params, steps=hist_steps, values=hist_vals
    )


def spsa_optimize(
    cost_fn,
    init_params,
    steps,
    stepsize,
    log_every=5,
    callback=None,
    *,
    keep_best=False,
    stepsize_decay=0.0,
    seed=0,
):
    """Two sampled evaluations per SPSA update, plus logged/selection costs.

    Use a_t = stepsize / (t+1)^0.602 / (1+stepsize_decay*t) and
    c_t = 0.1 / (t+1)^0.101 with seeded independent Rademacher perturbations.
    No autodifferentiation of samples or empirical quantiles is attempted.
    """
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if type(log_every) is not int or log_every < 1:
        raise ValueError("log_every must be a positive integer")
    if not onp.isfinite(stepsize) or stepsize <= 0:
        raise ValueError("stepsize must be finite and positive")
    if not onp.isfinite(stepsize_decay) or stepsize_decay < 0:
        raise ValueError("stepsize_decay must be finite and nonnegative")

    rng = onp.random.default_rng(seed)
    params = onp.array(init_params, dtype=float, copy=True)
    best_params, best_value = params.copy(), float("inf")

    def evaluate(theta):
        value = float(cost_fn(theta))
        if not onp.isfinite(value):
            raise ValueError("Training cost is nonfinite")
        return value

    if keep_best:
        best_value = evaluate(params)
    hist_steps, hist_vals = [], []
    for t in range(steps):
        delta = rng.choice([-1.0, 1.0], size=params.shape)
        perturbation = 0.1 / (t + 1) ** 0.101
        plus = evaluate(params + perturbation * delta)
        minus = evaluate(params - perturbation * delta)
        gradient = (plus - minus) * delta / (2 * perturbation)
        rate = stepsize / (t + 1) ** 0.602 / (1 + stepsize_decay * t)
        params = params - rate * gradient
        record = (t + 1) % log_every == 0 or t + 1 == steps
        if keep_best or record:
            value = evaluate(params)
            if value < best_value:
                best_params, best_value = params.copy(), value
        if record:
            hist_steps.append(t + 1)
            hist_vals.append(value)
            if callback is not None:
                callback(t + 1, value, params.copy())
    return OptimizeResult(
        params=best_params if keep_best else params, steps=hist_steps, values=hist_vals
    )
