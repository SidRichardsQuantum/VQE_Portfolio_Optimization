"""Variational quantum allocation, with legacy fractional-VQE API aliases.

The circuit parameterises weights; the mean-variance cost is evaluated
classically. This is not Hamiltonian-expectation VQE.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np

from .ansatz import apply_fractional_ansatz, fractional_ansatz_shape
from .metrics import symmetrize
from .optimize import adam_optimize, report_phase
from .types import (
    VariationalAllocatorConfig,
    VariationalAllocatorResult,
    LambdaSweepConfig,
    OptimizeTrace,
)
from .utils import set_global_seed

ALLOCATOR_SMOOTHING = 1e-12


def angles_to_weights(expvals_z: np.ndarray) -> np.ndarray:
    """Map physical Z expectations to a numerically feasible simplex allocation.

    For s_i=(1-z_i)/2 and epsilon=1e-12, use
    w_i=(s_i+epsilon/n)/(sum(s)+epsilon). Zero mass gives equal weights.
    Clip only numerical expectation overshoots within 1e-8. The normalisation
    is smooth in physical masses, with clipping at the measurement boundaries.
    Concrete input checks do not replace the differentiable calculation.
    """
    z = np.array(expvals_z)
    concrete = qml.math.toarray(z)
    if concrete.ndim != 1 or not concrete.size:
        raise ValueError("Expected a nonempty vector of Z expectations.")
    if np.iscomplexobj(concrete) or not np.all(np.isfinite(concrete)):
        raise ValueError("Z expectations must be finite and real.")
    if np.any(concrete < -1 - 1e-8) or np.any(concrete > 1 + 1e-8):
        raise ValueError("Z expectations must lie in [-1, 1].")
    s = (1.0 - np.clip(z, -1, 1)) * 0.5
    return (s + ALLOCATOR_SMOOTHING / len(z)) / (np.sum(s) + ALLOCATOR_SMOOTHING)


def run_variational_allocator(
    mu: np.ndarray,
    Sigma: np.ndarray,
    cfg: VariationalAllocatorConfig = VariationalAllocatorConfig(),
    *,
    candidate_callback=None,
    phase_callback=None,
) -> VariationalAllocatorResult:
    """Train a quantum-circuit allocation map using the classical portfolio cost."""
    set_global_seed(cfg.seed)

    mu = np.array(mu, requires_grad=False)
    Sigma = symmetrize(np.array(Sigma, requires_grad=False))
    n = len(mu)
    param_shape = fractional_ansatz_shape(cfg.ansatz, cfg.depth, n)

    dev = qml.device(cfg.device, wires=n)

    def ansatz(thetas: np.ndarray) -> None:
        apply_fractional_ansatz(cfg.ansatz, thetas, depth=cfg.depth, n_wires=n)

    @qml.qnode(dev, interface="autograd")
    def expvals_z(thetas: np.ndarray):
        ansatz(thetas)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    expvals_z = qml.set_shots(expvals_z, cfg.shots)

    def objective(thetas: np.ndarray):
        z = qml.math.stack(expvals_z(thetas))
        w = angles_to_weights(z)
        ret = qml.math.dot(mu, w)
        risk = qml.math.dot(w, qml.math.dot(Sigma, w))
        return -ret + cfg.lam * risk

    def report(step, value, params):
        if candidate_callback is not None:
            report_phase(phase_callback, "candidate_processing")
            candidate_callback(
                angles_to_weights(np.array(expvals_z(params))),
                {"step": step, "source": "allocation"},
            )
            report_phase(phase_callback, "optimization")

    init = np.array(np.random.uniform(0, np.pi, size=param_shape), requires_grad=True)
    report_phase(phase_callback, "optimization")
    opt_res = adam_optimize(
        objective,
        init,
        steps=cfg.steps,
        stepsize=cfg.stepsize,
        log_every=cfg.log_every,
        callback=report,
    )

    report_phase(phase_callback, "candidate_processing")
    z = np.array(expvals_z(opt_res.params))
    w = angles_to_weights(z)

    if candidate_callback is not None:
        candidate_callback(
            w, {"step": cfg.steps, "source": "allocation", "final": True}
        )
    return VariationalAllocatorResult(
        thetas=np.array(opt_res.params, requires_grad=False),
        cost_trace=OptimizeTrace(steps=opt_res.steps, values=opt_res.values),
        weights=np.array(w, requires_grad=False),
    )


def allocator_lambda_sweep(
    mu: np.ndarray,
    Sigma: np.ndarray,
    cfg: VariationalAllocatorConfig,
    sweep: LambdaSweepConfig,
) -> VariationalAllocatorResult:
    """
    Warm-start optional sweep over lambda values.
    """
    set_global_seed(cfg.seed)

    mu = np.array(mu, requires_grad=False)
    Sigma = symmetrize(np.array(Sigma, requires_grad=False))
    n = len(mu)
    param_shape = fractional_ansatz_shape(cfg.ansatz, cfg.depth, n)

    dev = qml.device(cfg.device, wires=n)

    def ansatz(thetas: np.ndarray) -> None:
        apply_fractional_ansatz(cfg.ansatz, thetas, depth=cfg.depth, n_wires=n)

    @qml.qnode(dev, interface="autograd")
    def expvals_z(thetas: np.ndarray):
        ansatz(thetas)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    expvals_z = qml.set_shots(expvals_z, cfg.shots)

    # Start point for sweep
    thetas = np.array(np.random.uniform(0, np.pi, size=param_shape), requires_grad=True)
    if sweep.warm_start:
        base = run_fractional_vqe(mu, Sigma, cfg)
        thetas = np.array(base.thetas, requires_grad=True)

    allocs = []
    lambdas = np.array(list(sweep.lambdas), dtype=float)

    for lam_val in lambdas:

        def objective(t: np.ndarray):
            z = qml.math.stack(expvals_z(t))
            w = angles_to_weights(z)
            ret = qml.math.dot(mu, w)
            risk = qml.math.dot(w, qml.math.dot(Sigma, w))
            return -ret + float(lam_val) * risk

        opt_res = adam_optimize(
            objective,
            thetas,
            steps=sweep.steps_per_lambda,
            stepsize=sweep.stepsize,
            log_every=max(sweep.steps_per_lambda, 1),
        )
        thetas = np.array(opt_res.params, requires_grad=True)  # warm start next lambda

        z = np.array(expvals_z(thetas))
        w = angles_to_weights(z)
        allocs.append(np.array(w, dtype=float))

    allocs_arr = np.vstack(allocs)

    # If we didn't already compute a base run for warm-start, compute it once now.
    if not sweep.warm_start:
        base = run_fractional_vqe(mu, Sigma, cfg)

    base_dict = dict(base.__dict__)
    base_dict.pop("lambdas", None)
    base_dict.pop("allocs_by_lambda", None)

    return VariationalAllocatorResult(
        **base_dict,
        lambdas=lambdas,
        allocs_by_lambda=allocs_arr,
    )


# Historical imports remain exact aliases, with no separate implementation.
run_fractional_vqe = run_variational_allocator
fractional_lambda_sweep = allocator_lambda_sweep

# Preserve historical type imports from this module as well.
FractionalVQEConfig = VariationalAllocatorConfig
FractionalVQEResult = VariationalAllocatorResult
