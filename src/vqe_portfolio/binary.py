from __future__ import annotations

from collections import Counter

import pennylane as qml
from pennylane import numpy as np

from .ansatz import apply_binary_ansatz, binary_ansatz_shape
from .metrics import symmetrize
from .reliability import resolve_penalty, energy_scale
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
    alpha = resolve_penalty(mu, Sigma, lam, k, alpha)

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


def _ordered_sample_counts(
    counts: Counter[tuple[int, ...]],
) -> dict[str, int]:
    """Return sample counts sorted by computational-basis bitstring order."""
    return {
        "".join(map(str, bitstring)): int(count)
        for bitstring, count in sorted(counts.items())
    }


def run_binary_vqe(
    mu: np.ndarray,
    Sigma: np.ndarray,
    cfg: BinaryVQEConfig = BinaryVQEConfig(),
    *,
    candidate_callback=None,
) -> BinaryVQEResult:
    """Train and sample Binary VQE, with optional feasible-candidate reporting.

    Each callback checkpoint includes marginal measurement, sampling, and Top-K
    fallback when no sampled portfolio is feasible. Their overhead is part of
    execution. Restarts select by penalized training energy, without references.
    """
    set_global_seed(cfg.seed)
    mu = np.array(mu, requires_grad=False)
    Sigma = symmetrize(np.array(Sigma, requires_grad=False))
    n = len(mu)
    if not (1 <= cfg.k <= n):
        raise ValueError(f"cfg.k must be in [1, {n}] but got {cfg.k}")
    if type(cfg.restarts) is not int or cfg.restarts < 1:
        raise ValueError("restarts must be a positive integer")
    if type(cfg.shots_sample) is not int or cfg.shots_sample < 1:
        raise ValueError("shots_sample must be a positive integer")
    param_shape = binary_ansatz_shape(cfg.ansatz, cfg.depth, n)
    if cfg.depth < 1:
        raise ValueError("depth must be positive")
    alpha = resolve_penalty(mu, Sigma, cfg.lam, cfg.k, cfg.alpha)
    H = build_ising_hamiltonian(mu, Sigma, cfg.lam, alpha, cfg.k)
    scale = energy_scale(H) if cfg.normalize_energy else 1.0
    dev_train = qml.device(cfg.device, wires=n)

    def ansatz(params):
        apply_binary_ansatz(cfg.ansatz, params, depth=cfg.depth, n_wires=n)

    @qml.qnode(dev_train, interface="autograd")
    def energy(params):
        ansatz(params)
        return qml.expval(H)

    energy = qml.set_shots(energy, cfg.shots_train)

    def training_cost(params):
        return energy(params) / scale

    # Preserve the historical first initialization before sampling-device setup.
    first_init = np.array(
        np.random.uniform(0, np.pi, size=param_shape), requires_grad=True
    )

    @qml.qnode(dev_train, interface="autograd")
    def exp_z(params):
        ansatz(params)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    exp_z = qml.set_shots(exp_z, cfg.shots_train)
    dev_samp = qml.device(cfg.device, wires=n)

    @qml.qnode(dev_samp)
    def sample_bits(params):
        ansatz(params)
        return qml.sample(wires=range(n))

    sample_bits = qml.set_shots(sample_bits, cfg.shots_sample)

    def measure(params):
        z = np.stack(exp_z(params))
        x_prob = selection_prob_from_z(z)
        x_topk = topk_project(x_prob, cfg.k)
        rows = np.array(sample_bits(params)).reshape(cfg.shots_sample, n)
        counts = Counter(tuple(map(int, row)) for row in rows)
        feasible = [bs for bs in counts if sum(bs) == cfg.k]
        best = None
        if feasible:
            best = np.array(
                min(
                    feasible,
                    key=lambda bs: float(
                        cfg.lam * np.array(bs) @ Sigma @ np.array(bs)
                        - mu @ np.array(bs)
                    ),
                ),
                dtype=int,
            )
        return z, x_prob, x_topk, counts, best

    restart_energies = []
    selected, selected_value = None, float("inf")
    for restart in range(cfg.restarts):
        init = (
            first_init
            if restart == 0
            else np.array(
                np.random.uniform(0, np.pi, size=param_shape), requires_grad=True
            )
        )

        def report(step, value, params):
            if candidate_callback is not None:
                _, _, topk, _, best = measure(params)
                candidate_callback(
                    topk if best is None else best,
                    {
                        "step": step,
                        "restart": restart,
                        "source": "topk_fallback" if best is None else "sample",
                    },
                )

        opt = adam_optimize(
            training_cost,
            init,
            steps=cfg.steps,
            stepsize=cfg.stepsize,
            log_every=cfg.log_every,
            callback=report,
            keep_best=cfg.keep_best,
            stepsize_decay=cfg.stepsize_decay,
        )
        value = float(energy(opt.params))
        restart_energies.append(value)
        if value < selected_value:
            selected, selected_value = opt, value
    opt_res = selected
    z, x_prob, x_topk, counts, best = measure(opt_res.params)
    x_mode = np.array(max(counts, key=counts.get), dtype=int)
    if candidate_callback is not None:
        candidate_callback(
            x_topk if best is None else best,
            {
                "step": cfg.steps,
                "source": "topk_fallback" if best is None else "sample",
                "final": True,
            },
        )
    return BinaryVQEResult(
        params=np.array(opt_res.params, requires_grad=False),
        energy_trace=OptimizeTrace(
            steps=opt_res.steps, values=[v * scale for v in opt_res.values]
        ),
        z_expect=np.array(z, requires_grad=False),
        x_prob=np.array(x_prob, requires_grad=False),
        x_round=(x_prob >= 0.5).astype(int),
        x_topk=x_topk,
        sample_counts=_ordered_sample_counts(counts),
        x_mode=x_mode,
        x_best_feasible=best,
        effective_alpha=alpha,
        energy_scale=scale,
        restart_energies=tuple(restart_energies),
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
    param_shape = binary_ansatz_shape(cfg.ansatz, cfg.depth, n)

    dev = qml.device(cfg.device, wires=n)

    def ansatz(params: np.ndarray) -> None:
        apply_binary_ansatz(cfg.ansatz, params, depth=cfg.depth, n_wires=n)

    probs = []
    lambdas = np.array(list(sweep.lambdas), dtype=float)
    if lambdas.ndim != 1 or not lambdas.size:
        raise ValueError("sweep.lambdas must be a non-empty 1D sequence")

    for lam_val in lambdas:
        H = build_ising_hamiltonian(mu, Sigma, float(lam_val), cfg.alpha, cfg.k)

        @qml.qnode(dev, interface="autograd")
        def energy(params: np.ndarray):
            ansatz(params)
            return qml.expval(H)

        energy = qml.set_shots(energy, cfg.shots_train)

        scale = energy_scale(H) if cfg.normalize_energy else 1.0
        best_value, opt_res = float("inf"), None
        if type(cfg.restarts) is not int or cfg.restarts < 1:
            raise ValueError("restarts must be a positive integer")
        for _ in range(cfg.restarts):
            init = np.array(
                np.random.uniform(0, np.pi, size=param_shape), requires_grad=True
            )
            result = adam_optimize(
                lambda params: energy(params) / scale,
                init,
                steps=sweep.steps_per_lambda,
                stepsize=sweep.stepsize,
                log_every=max(sweep.steps_per_lambda, 1),
                keep_best=cfg.keep_best,
                stepsize_decay=cfg.stepsize_decay,
            )
            value = float(energy(result.params))
            if value < best_value:
                best_value, opt_res = value, result

        @qml.qnode(dev, interface="autograd")
        def exp_z(params: np.ndarray):
            ansatz(params)
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        exp_z = qml.set_shots(exp_z, cfg.shots_train)

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
