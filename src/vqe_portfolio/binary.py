from __future__ import annotations

from collections import Counter
from typing import Optional

import pennylane as qml
from pennylane import numpy as np

from .ansatz import binary_hwe_ry_cz_ring
from .metrics import symmetrize
from .optimize import adam_optimize
from .types import BinaryVQEConfig, BinaryVQEResult, LambdaSweepConfig, OptimizeTrace
from .utils import topk_onehot
from .utils import set_global_seed


def build_ising_hamiltonian(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    alpha: float,
    k: int,
) -> qml.Hamiltonian:
    """
    Build Ising Hamiltonian for:
        f(x) = lam x^T Sigma x - mu^T x + alpha (sum x - k)^2
    using x_i = (1 - Z_i)/2.

    Returns PennyLane Hamiltonian H such that <H> corresponds to the expanded Ising objective.
    """
    mu = np.array(mu, requires_grad=False)
    Sigma = np.array(Sigma, requires_grad=False)
    Sigma = symmetrize(Sigma)
    n = len(mu)

    if not (1 <= k <= n):
        raise ValueError(f"k must be in [1, {n}] but got {k}")

    # QUBO constants
    const = float(alpha * k**2)

    # linear a_i x_i
    a = lam * np.diag(Sigma) - mu + alpha * (1 - 2 * k)

    # pairwise b_ij x_i x_j for i<j only
    b = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            b[i, j] = 2 * lam * Sigma[i, j] + 2 * alpha

    # QUBO -> Ising coefficients
    c = const
    h = np.zeros(n)
    J = np.zeros((n, n))

    for i in range(n):
        c += 0.5 * a[i]
        h[i] += -0.5 * a[i]

    for i in range(n):
        for j in range(i + 1, n):
            bij = b[i, j]
            c += 0.25 * bij
            h[i] += -0.25 * bij
            h[j] += -0.25 * bij
            J[i, j] += 0.25 * bij

    coeffs = [float(c)]
    ops = [qml.Identity(0)]

    for i in range(n):
        if abs(h[i]) > 1e-12:
            coeffs.append(float(h[i]))
            ops.append(qml.PauliZ(i))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                coeffs.append(float(J[i, j]))
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    return qml.Hamiltonian(coeffs, ops)


def selection_prob_from_z(z: np.ndarray) -> np.ndarray:
    """x_prob = (1 - <Z>)/2 in [0,1]."""
    return (1.0 - np.array(z)) * 0.5


def topk_project(x_prob: np.ndarray, k: int) -> np.ndarray:
    """Project probabilities to a binary Top-K selection (dtype=int)."""
    return topk_onehot(x_prob, k).astype(int)


def run_binary_vqe(
    mu: np.ndarray,
    Sigma: np.ndarray,
    cfg: BinaryVQEConfig = BinaryVQEConfig(),
) -> BinaryVQEResult:
    """
    Train VQE on the binary selection Ising Hamiltonian, then compute:
    - marginal inclusion probabilities
    - naive rounding
    - Top-K projection
    - samples + mode + best feasible sampled
    """
 
    set_global_seed(cfg.seed)

    mu = np.array(mu, requires_grad=False)
    Sigma = np.array(Sigma, requires_grad=False)
    Sigma = symmetrize(Sigma)
    n = len(mu)

    if not (1 <= cfg.k <= n):
        raise ValueError(f"cfg.k must be in [1, {n}] but got {cfg.k}")

    H = build_ising_hamiltonian(mu, Sigma, cfg.lam, cfg.alpha, cfg.k)

    dev_train = qml.device(cfg.device, wires=n, shots=cfg.shots_train)

    def ansatz(params: np.ndarray) -> None:
        binary_hwe_ry_cz_ring(params, depth=cfg.depth, n_wires=n)

    @qml.qnode(dev_train, interface="autograd")
    def energy(params: np.ndarray):
        ansatz(params)
        return qml.expval(H)

    init = np.array(np.random.uniform(0, np.pi, size=(cfg.depth, n)), requires_grad=True)
    opt_res = adam_optimize(
        energy, init, steps=cfg.steps, stepsize=cfg.stepsize, log_every=cfg.log_every
    )

    @qml.qnode(dev_train, interface="autograd")
    def exp_z(params: np.ndarray):
        ansatz(params)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    z = np.stack(exp_z(opt_res.params))
    x_prob = selection_prob_from_z(z)
    x_round = (x_prob >= 0.5).astype(int)
    x_topk = topk_project(x_prob, cfg.k)

    dev_samp = qml.device(cfg.device, wires=n, shots=cfg.shots_sample)

    @qml.qnode(dev_samp)
    def sample_bits(params: np.ndarray):
        ansatz(params)
        return qml.sample(wires=range(n))

    samples = sample_bits(opt_res.params)
    rows = np.array(samples)
    counts: Counter[tuple[int, ...]] = Counter(tuple(map(int, row)) for row in rows)

    mode_bitstring = max(counts, key=counts.get)
    x_mode = np.array(mode_bitstring, dtype=int)

    def objective_value(x) -> float:
        x = np.array(x, dtype=float)
        return float(cfg.lam * x @ Sigma @ x - mu @ x + cfg.alpha * (x.sum() - cfg.k) ** 2)

    feasible = [bs for bs in counts if sum(bs) == cfg.k]
    if feasible:
        best = min(feasible, key=lambda bs: objective_value(bs))
        x_best_feasible: Optional[np.ndarray] = np.array(best, dtype=int)
    else:
        x_best_feasible = None

    return BinaryVQEResult(
        params=np.array(opt_res.params, requires_grad=False),
        energy_trace=OptimizeTrace(steps=opt_res.steps, values=opt_res.values),
        z_expect=np.array(z, requires_grad=False),
        x_prob=np.array(x_prob, requires_grad=False),
        x_round=np.array(x_round, requires_grad=False),
        x_topk=np.array(x_topk, requires_grad=False),
        sample_counts={"".join(map(str, k)): int(v) for k, v in counts.items()},
        x_mode=np.array(x_mode, requires_grad=False),
        x_best_feasible=x_best_feasible,
    )


def binary_lambda_sweep(
    mu: np.ndarray,
    Sigma: np.ndarray,
    cfg: BinaryVQEConfig,
    sweep: LambdaSweepConfig,
) -> BinaryVQEResult:
    """
    Re-optimize for each lambda and return probs_by_lambda.
    Matches notebook behavior (fresh init each lambda).
    """
    set_global_seed(cfg.seed)

    mu = np.array(mu, requires_grad=False)
    Sigma = symmetrize(np.array(Sigma, requires_grad=False))
    n = len(mu)

    if not (1 <= cfg.k <= n):
        raise ValueError(f"cfg.k must be in [1, {n}] but got {cfg.k}")

    dev = qml.device(cfg.device, wires=n, shots=cfg.shots_train)

    def ansatz(params: np.ndarray) -> None:
        binary_hwe_ry_cz_ring(params, depth=cfg.depth, n_wires=n)

    probs = []
    lambdas = np.array(list(sweep.lambdas), dtype=float)

    for lam_val in lambdas:
        H = build_ising_hamiltonian(mu, Sigma, float(lam_val), cfg.alpha, cfg.k)

        @qml.qnode(dev, interface="autograd")
        def energy(params: np.ndarray):
            ansatz(params)
            return qml.expval(H)

        init = np.array(np.random.uniform(0, np.pi, size=(cfg.depth, n)), requires_grad=True)
        opt_res = adam_optimize(
            energy,
            init,
            steps=sweep.steps_per_lambda,
            stepsize=sweep.stepsize,
            log_every=max(sweep.steps_per_lambda, 1),
        )

        @qml.qnode(dev, interface="autograd")
        def exp_z(params: np.ndarray):
            ansatz(params)
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        z = np.stack(exp_z(opt_res.params))
        probs.append(selection_prob_from_z(z))

    probs_arr = np.vstack(probs)

    base = run_binary_vqe(mu, Sigma, cfg)
    base_dict = dict(base.__dict__)
    base_dict.pop("lambdas", None)
    base_dict.pop("probs_by_lambda", None)

    return BinaryVQEResult(
        **base_dict,
        lambdas=lambdas,
        probs_by_lambda=probs_arr,
    )
