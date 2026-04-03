from __future__ import annotations

from collections import Counter
from typing import Optional

import pennylane as qml
from pennylane import numpy as np

from .binary import build_ising_hamiltonian
from .metrics import symmetrize
from .optimize import adam_optimize
from .types import LambdaSweepConfig, OptimizeTrace, QAOAConfig, QAOAResult
from .utils import set_global_seed
from .utils import topk_onehot


def _bitstring_array_from_index(index: int, n: int) -> np.ndarray:
    bits = np.array(list(np.binary_repr(index, width=n)), dtype=int)
    return bits


def _sample_counts_to_counter(samples: np.ndarray) -> Counter[tuple[int, ...]]:
    rows = np.array(samples)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return Counter(tuple(map(int, row)) for row in rows)


def _selection_prob_from_probs(probs: np.ndarray, n: int) -> np.ndarray:
    x_prob = np.zeros(n, dtype=float)
    for idx, p in enumerate(np.array(probs, dtype=float)):
        bits = _bitstring_array_from_index(idx, n)
        x_prob += float(p) * bits
    return x_prob


def _topk_project(x_prob: np.ndarray, k: int) -> np.ndarray:
    return topk_onehot(x_prob, k).astype(int)


def _objective_value(
    x: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    alpha: float,
    k: int,
) -> float:
    x = np.array(x, dtype=float)
    return float(lam * x @ Sigma @ x - mu @ x + alpha * (x.sum() - k) ** 2)


def _prepare_initial_state(n: int, k: int, mixer: str) -> None:
    if mixer == "x":
        for i in range(n):
            qml.Hadamard(wires=i)
        return

    if mixer == "xy":
        basis = np.zeros(n, dtype=int)
        basis[:k] = 1
        qml.BasisState(basis, wires=range(n))
        return

    raise ValueError(f"Unsupported mixer '{mixer}'. Expected 'x' or 'xy'.")


def _apply_x_mixer(beta: float, n: int) -> None:
    for i in range(n):
        qml.RX(2.0 * beta, wires=i)


def _apply_xy_mixer(beta: float, n: int) -> None:
    if n <= 1:
        return

    edges_even = [(i, i + 1) for i in range(0, n - 1, 2)]
    edges_odd = [(i, i + 1) for i in range(1, n - 1, 2)]

    for i, j in edges_even:
        qml.IsingXX(2.0 * beta, wires=[i, j])
        qml.IsingYY(2.0 * beta, wires=[i, j])

    for i, j in edges_odd:
        qml.IsingXX(2.0 * beta, wires=[i, j])
        qml.IsingYY(2.0 * beta, wires=[i, j])

    if n > 2:
        qml.IsingXX(2.0 * beta, wires=[n - 1, 0])
        qml.IsingYY(2.0 * beta, wires=[n - 1, 0])


def _apply_mixer(beta: float, n: int, mixer: str) -> None:
    if mixer == "x":
        _apply_x_mixer(beta, n)
        return
    if mixer == "xy":
        _apply_xy_mixer(beta, n)
        return
    raise ValueError(f"Unsupported mixer '{mixer}'. Expected 'x' or 'xy'.")


def _qaoa_layer(
    gamma: float, beta: float, H_cost: qml.Hamiltonian, n: int, mixer: str
) -> None:
    qml.ApproxTimeEvolution(H_cost, gamma, 1)
    _apply_mixer(beta, n, mixer)


def _qaoa_ansatz(
    params: np.ndarray,
    H_cost: qml.Hamiltonian,
    n: int,
    k: int,
    mixer: str,
) -> None:
    _prepare_initial_state(n=n, k=k, mixer=mixer)

    gammas = params[:, 0]
    betas = params[:, 1]

    for gamma, beta in zip(gammas, betas):
        _qaoa_layer(gamma, beta, H_cost, n, mixer)


def _random_init(p: int, mixer: str) -> np.ndarray:
    gamma_hi = np.pi
    beta_hi = np.pi if mixer == "xy" else 0.5 * np.pi

    params = np.zeros((p, 2), dtype=float)
    params[:, 0] = np.random.uniform(0.0, gamma_hi, size=p)
    params[:, 1] = np.random.uniform(0.0, beta_hi, size=p)
    return np.array(params, requires_grad=True)


def _validate_inputs(
    mu: np.ndarray, Sigma: np.ndarray, cfg: QAOAConfig
) -> tuple[np.ndarray, np.ndarray, int]:
    mu = np.array(mu, requires_grad=False)
    Sigma = np.array(Sigma, requires_grad=False)
    Sigma = symmetrize(Sigma)

    if mu.ndim != 1:
        raise ValueError(f"mu must be 1D; got shape {mu.shape}")
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"Sigma must be square; got shape {Sigma.shape}")
    if Sigma.shape[0] != mu.shape[0]:
        raise ValueError(
            f"Sigma shape {Sigma.shape} is incompatible with mu shape {mu.shape}"
        )

    n = len(mu)

    if not (1 <= cfg.k <= n):
        raise ValueError(f"cfg.k must be in [1, {n}] but got {cfg.k}")
    if cfg.p <= 0:
        raise ValueError(f"cfg.p must be positive; got {cfg.p}")
    if cfg.steps <= 0:
        raise ValueError(f"cfg.steps must be positive; got {cfg.steps}")
    if cfg.stepsize <= 0:
        raise ValueError(f"cfg.stepsize must be positive; got {cfg.stepsize}")
    if cfg.mixer not in {"x", "xy"}:
        raise ValueError(f"cfg.mixer must be 'x' or 'xy'; got {cfg.mixer!r}")

    return mu, Sigma, n


def run_qaoa(
    mu: np.ndarray,
    Sigma: np.ndarray,
    cfg: QAOAConfig = QAOAConfig(),
) -> QAOAResult:
    set_global_seed(cfg.seed)

    mu, Sigma, n = _validate_inputs(mu, Sigma, cfg)
    H_cost = build_ising_hamiltonian(mu, Sigma, cfg.lam, cfg.alpha, cfg.k)

    dev_train = qml.device(cfg.device, wires=n)

    @qml.qnode(dev_train, interface="autograd")
    def energy(params: np.ndarray):
        _qaoa_ansatz(params, H_cost=H_cost, n=n, k=cfg.k, mixer=cfg.mixer)
        return qml.expval(H_cost)

    energy = qml.set_shots(energy, cfg.shots_train)

    init = _random_init(cfg.p, cfg.mixer)
    opt_res = adam_optimize(
        energy,
        init,
        steps=cfg.steps,
        stepsize=cfg.stepsize,
        log_every=cfg.log_every,
    )

    best_params = np.array(opt_res.params, requires_grad=False)
    gammas = np.array(best_params[:, 0], requires_grad=False)
    betas = np.array(best_params[:, 1], requires_grad=False)

    dev_prob = qml.device(cfg.device, wires=n)

    @qml.qnode(dev_prob)
    def state_probs(params: np.ndarray):
        _qaoa_ansatz(params, H_cost=H_cost, n=n, k=cfg.k, mixer=cfg.mixer)
        return qml.probs(wires=range(n))

    probs = np.array(state_probs(best_params), requires_grad=False)
    x_prob = _selection_prob_from_probs(probs, n)
    x_round = (x_prob >= 0.5).astype(int)
    x_topk = _topk_project(x_prob, cfg.k)

    dev_samp = qml.device(cfg.device, wires=n)

    @qml.qnode(dev_samp)
    def sample_bits(params: np.ndarray):
        _qaoa_ansatz(params, H_cost=H_cost, n=n, k=cfg.k, mixer=cfg.mixer)
        return qml.sample(wires=range(n))

    sample_bits = qml.set_shots(sample_bits, cfg.shots_sample)

    samples = sample_bits(best_params)
    counts = _sample_counts_to_counter(samples)

    mode_bitstring = max(counts, key=counts.get)
    x_mode = np.array(mode_bitstring, dtype=int)

    feasible = [bs for bs in counts if sum(bs) == cfg.k]
    if feasible:
        best = min(
            feasible,
            key=lambda bs: _objective_value(
                np.array(bs, dtype=int),
                mu=mu,
                Sigma=Sigma,
                lam=cfg.lam,
                alpha=cfg.alpha,
                k=cfg.k,
            ),
        )
        x_best_feasible: Optional[np.ndarray] = np.array(best, dtype=int)
    else:
        x_best_feasible = None

    return QAOAResult(
        params=best_params,
        gammas=gammas,
        betas=betas,
        cost_trace=OptimizeTrace(steps=opt_res.steps, values=opt_res.values),
        state_probs=probs,
        x_prob=np.array(x_prob, requires_grad=False),
        x_round=np.array(x_round, requires_grad=False),
        x_topk=np.array(x_topk, requires_grad=False),
        sample_counts={"".join(map(str, k)): int(v) for k, v in counts.items()},
        x_mode=np.array(x_mode, requires_grad=False),
        x_best_feasible=x_best_feasible,
    )


def qaoa_lambda_sweep(
    mu: np.ndarray,
    Sigma: np.ndarray,
    cfg: QAOAConfig,
    sweep: LambdaSweepConfig,
) -> QAOAResult:
    set_global_seed(cfg.seed)

    mu, Sigma, n = _validate_inputs(mu, Sigma, cfg)
    lambdas = np.array(list(sweep.lambdas), dtype=float)

    if lambdas.ndim != 1 or lambdas.size == 0:
        raise ValueError("sweep.lambdas must be a non-empty 1D sequence")

    dev = qml.device(cfg.device, wires=n)
    probs_by_lambda = []
    params = None

    for idx, lam_val in enumerate(lambdas):
        H_cost = build_ising_hamiltonian(mu, Sigma, float(lam_val), cfg.alpha, cfg.k)

        @qml.qnode(dev, interface="autograd")
        def energy(theta: np.ndarray):
            _qaoa_ansatz(theta, H_cost=H_cost, n=n, k=cfg.k, mixer=cfg.mixer)
            return qml.expval(H_cost)

        energy = qml.set_shots(energy, cfg.shots_train)

        if idx == 0 or not sweep.warm_start or params is None:
            init = _random_init(cfg.p, cfg.mixer)
        else:
            init = np.array(params, requires_grad=True)

        opt_res = adam_optimize(
            energy,
            init,
            steps=sweep.steps_per_lambda,
            stepsize=sweep.stepsize,
            log_every=max(sweep.steps_per_lambda, 1),
        )
        params = np.array(opt_res.params, requires_grad=False)

        @qml.qnode(dev)
        def state_probs(theta: np.ndarray):
            _qaoa_ansatz(theta, H_cost=H_cost, n=n, k=cfg.k, mixer=cfg.mixer)
            return qml.probs(wires=range(n))

        probs = np.array(state_probs(params), dtype=float)
        probs_by_lambda.append(_selection_prob_from_probs(probs, n))

    probs_arr = np.vstack(probs_by_lambda)
    base = run_qaoa(mu, Sigma, cfg)
    base_dict = dict(base.__dict__)
    base_dict.pop("lambdas", None)
    base_dict.pop("probs_by_lambda", None)

    return QAOAResult(
        **base_dict,
        lambdas=lambdas,
        probs_by_lambda=probs_arr,
    )
