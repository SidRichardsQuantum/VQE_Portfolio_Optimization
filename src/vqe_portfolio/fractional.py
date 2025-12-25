from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np

from .ansatz import fractional_ry_layer
from .metrics import symmetrize
from .optimize import adam_optimize
from .types import FractionalVQEConfig, FractionalVQEResult, LambdaSweepConfig, OptimizeTrace
from .utils import set_global_seed

def angles_to_weights(expvals_z: np.ndarray) -> np.ndarray:
    """
    Given z_i = <Z_i>, map to simplex:
      s_i = (1 - z_i)/2 in [0,1]
      w_i = s_i / sum(s)
    """
    z = np.array(expvals_z)
    s = (1.0 - z) * 0.5
    denom = np.sum(s) + 1e-12
    return s / denom


def run_fractional_vqe(
    mu: np.ndarray,
    Sigma: np.ndarray,
    cfg: FractionalVQEConfig = FractionalVQEConfig(),
) -> FractionalVQEResult:
    set_global_seed(cfg.seed)

    mu = np.array(mu, requires_grad=False)
    Sigma = symmetrize(np.array(Sigma, requires_grad=False))
    n = len(mu)

    dev = qml.device(cfg.device, wires=n, shots=cfg.shots)

    def ansatz(thetas: np.ndarray) -> None:
        fractional_ry_layer(thetas, n_wires=n)

    @qml.qnode(dev, interface="autograd")
    def expvals_z(thetas: np.ndarray):
        ansatz(thetas)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    def objective(thetas: np.ndarray):
        z = qml.math.stack(expvals_z(thetas))
        w = angles_to_weights(z)
        ret = qml.math.dot(mu, w)
        risk = qml.math.dot(w, qml.math.dot(Sigma, w))
        return -ret + cfg.lam * risk

    init = np.array(np.random.uniform(0, np.pi, n), requires_grad=True)
    opt_res = adam_optimize(objective, init, steps=cfg.steps, stepsize=cfg.stepsize, log_every=cfg.log_every)

    z = np.array(expvals_z(opt_res.params))
    w = angles_to_weights(z)

    return FractionalVQEResult(
        thetas=np.array(opt_res.params, requires_grad=False),
        cost_trace=OptimizeTrace(steps=opt_res.steps, values=opt_res.values),
        weights=np.array(w, requires_grad=False),
    )


def fractional_lambda_sweep(
    mu: np.ndarray,
    Sigma: np.ndarray,
    cfg: FractionalVQEConfig,
    sweep: LambdaSweepConfig,
) -> FractionalVQEResult:
    """
    Warm-start optional sweep over lambda values.
    """
    set_global_seed(cfg.seed)

    mu = np.array(mu, requires_grad=False)
    Sigma = symmetrize(np.array(Sigma, requires_grad=False))
    n = len(mu)

    dev = qml.device(cfg.device, wires=n, shots=cfg.shots)

    def ansatz(thetas: np.ndarray) -> None:
        fractional_ry_layer(thetas, n_wires=n)

    @qml.qnode(dev, interface="autograd")
    def expvals_z(thetas: np.ndarray):
        ansatz(thetas)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    # Start point for sweep
    thetas = np.array(np.random.uniform(0, np.pi, n), requires_grad=True)
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

    return FractionalVQEResult(
        **base_dict,
        lambdas=lambdas,
        allocs_by_lambda=allocs_arr,
    )
