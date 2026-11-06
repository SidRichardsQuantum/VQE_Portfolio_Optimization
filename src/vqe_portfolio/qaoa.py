from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import comb
import time
from typing import Optional

import pennylane as qml
from pennylane import numpy as np

from .binary import build_ising_hamiltonian
from .cvar import (
    make_training_cost,
    optimize_training,
    training_metadata,
    validate_training_objective,
)
from .metrics import symmetrize
from .optimize import report_phase
from .types import LambdaSweepConfig, OptimizeTrace, QAOAConfig, QAOAResult
from .utils import set_global_seed
from .utils import topk_onehot


def _bitstring_array_from_index(index: int, n: int) -> np.ndarray:
    bits = np.array(list(np.binary_repr(index, width=n)), dtype=int)
    return bits


def _sample_counts_to_counter(samples: np.ndarray, n=None) -> Counter[tuple[int, ...]]:
    rows = np.array(samples)
    if rows.ndim == 1:
        rows = rows.reshape(-1, n) if n is not None else rows.reshape(1, -1)
    return Counter(tuple(map(int, row)) for row in rows)


def _ordered_sample_counts(
    counts: Counter[tuple[int, ...]],
) -> dict[str, int]:
    """Return sample counts sorted by computational-basis bitstring order."""
    return {
        "".join(map(str, bitstring)): int(count)
        for bitstring, count in sorted(counts.items())
    }


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


def _prepare_initial_state(
    n: int, k: int, mixer: str, xy_initial_state="basis"
) -> None:
    if mixer == "x":
        for i in range(n):
            qml.Hadamard(wires=i)
        return

    if mixer == "xy":
        if xy_initial_state == "dicke":
            amplitudes = np.zeros(2**n, requires_grad=False)
            for selected in combinations(range(n), k):
                index = sum(1 << (n - 1 - i) for i in selected)
                amplitudes[index] = 1 / np.sqrt(comb(n, k))
            qml.StatePrep(amplitudes, wires=range(n))
            return
        if xy_initial_state != "basis":
            raise ValueError("xy_initial_state must be 'basis' or 'dicke'")
        basis = np.zeros(n, dtype=int)
        basis[:k] = 1
        qml.BasisState(basis, wires=range(n))
        return

    raise ValueError(f"Unsupported mixer '{mixer}'. Expected 'x' or 'xy'.")


def _apply_x_mixer(beta: float, n: int) -> None:
    for i in range(n):
        qml.RX(2.0 * beta, wires=i)


def xy_mixer_edges(n: int, topology="ring") -> tuple[tuple[int, int], ...]:
    """Ordered exchange edges shared by execution and reachability analysis."""
    if topology not in {"ring", "complete"}:
        raise ValueError("xy_topology must be 'ring' or 'complete'")
    if topology == "complete":
        return tuple(combinations(range(n), 2))
    if n <= 1:
        return ()

    edges_even = [(i, i + 1) for i in range(0, n - 1, 2)]
    edges_odd = [(i, i + 1) for i in range(1, n - 1, 2)]

    return tuple(edges_even + edges_odd + ([(n - 1, 0)] if n > 2 else []))


def _apply_xy_mixer(beta: float, n: int, topology="ring") -> None:
    """Ordered XX/YY exchange gates preserving ideal Hamming weight exactly.

    Overlapping edges need not commute: this is an alternating-operator mixer,
    not generally the exact exponential of their summed Hamiltonian.
    """
    for i, j in xy_mixer_edges(n, topology):
        qml.IsingXX(2.0 * beta, wires=[i, j])
        qml.IsingYY(2.0 * beta, wires=[i, j])


def _apply_mixer(beta: float, n: int, mixer: str, xy_topology="ring") -> None:
    if mixer == "x":
        _apply_x_mixer(beta, n)
        return
    if mixer == "xy":
        _apply_xy_mixer(beta, n, xy_topology)
        return
    raise ValueError(f"Unsupported mixer '{mixer}'. Expected 'x' or 'xy'.")


def _qaoa_layer(
    gamma: float,
    beta: float,
    H_cost: qml.Hamiltonian,
    n: int,
    mixer: str,
    xy_topology="ring",
) -> None:
    qml.ApproxTimeEvolution(H_cost, gamma, 1)
    _apply_mixer(beta, n, mixer, xy_topology)


def _qaoa_ansatz(
    params: np.ndarray,
    H_cost: qml.Hamiltonian,
    n: int,
    k: int,
    mixer: str,
    xy_initial_state="basis",
    xy_topology="ring",
) -> None:
    _prepare_initial_state(n=n, k=k, mixer=mixer, xy_initial_state=xy_initial_state)

    gammas = params[:, 0]
    betas = params[:, 1]

    for gamma, beta in zip(gammas, betas):
        _qaoa_layer(gamma, beta, H_cost, n, mixer, xy_topology)


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
    validate_training_objective(cfg)
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
    if type(cfg.steps) is not int or cfg.steps < 0:
        raise ValueError(f"cfg.steps must be a nonnegative integer; got {cfg.steps}")
    if cfg.stepsize <= 0:
        raise ValueError(f"cfg.stepsize must be positive; got {cfg.stepsize}")
    if cfg.mixer not in {"x", "xy"}:
        raise ValueError(f"cfg.mixer must be 'x' or 'xy'; got {cfg.mixer!r}")

    if cfg.xy_initial_state not in {"basis", "dicke"}:
        raise ValueError("xy_initial_state must be 'basis' or 'dicke'")
    if cfg.xy_topology not in {"ring", "complete"}:
        raise ValueError("xy_topology must be 'ring' or 'complete'")
    if cfg.mixer == "x" and (
        cfg.xy_initial_state != "basis" or cfg.xy_topology != "ring"
    ):
        raise ValueError("XY state/topology options require mixer='xy'")
    if type(cfg.shots_sample) is not int or cfg.shots_sample < 1:
        raise ValueError("shots_sample must be a positive integer")
    return mu, Sigma, n


def _sampling_qnode(H_cost, n, cfg):
    dev = qml.device(cfg.device, wires=n)

    @qml.qnode(dev)
    def sample_bits(params):
        _qaoa_ansatz(
            params, H_cost, n, cfg.k, cfg.mixer, cfg.xy_initial_state, cfg.xy_topology
        )
        return qml.sample(wires=range(n))

    return qml.set_shots(sample_bits, cfg.shots_sample)


def _sample_initial_qaoa(
    mu, Sigma, cfg, deadline, candidate_callback, phase_callback=None
):
    """Repeated circuit batches at the production random initialization.

    No training, reference scores, exact probabilities, or classical sampling
    shortcuts are used. Setup and every circuit batch consume the allowance.
    """
    if cfg.steps != 0:
        raise ValueError("Untrained sampling requires steps=0")
    set_global_seed(cfg.seed)
    mu, Sigma, n = _validate_inputs(mu, Sigma, cfg)
    h = build_ising_hamiltonian(mu, Sigma, cfg.lam, cfg.alpha, cfg.k)
    # Match production device creation before random parameter initialization.
    qml.device(cfg.device, wires=n)
    params = _random_init(cfg.p, cfg.mixer)
    return _sample_qaoa_parameters(
        mu,
        Sigma,
        cfg,
        params,
        deadline,
        candidate_callback,
        phase_callback,
        source="untrained_sample",
        h=h,
    )


def _sample_qaoa_parameters(
    mu,
    Sigma,
    cfg,
    params,
    deadline,
    candidate_callback,
    phase_callback=None,
    *,
    source="trained_sample",
    h=None,
):
    """Spend remaining time on actual circuit batches at frozen parameters."""
    n = len(mu)
    if h is None:
        h = build_ising_hamiltonian(mu, Sigma, cfg.lam, cfg.alpha, cfg.k)
    sample_bits = _sampling_qnode(h, n, cfg)
    report_phase(phase_callback, "candidate_processing")
    batches = 0
    seen = set()
    while time.perf_counter() < deadline:
        counts = _sample_counts_to_counter(sample_bits(params), n)
        feasible = [bs for bs in counts if sum(bs) == cfg.k]
        seen.update(feasible)
        batches += 1
        if feasible:
            best = min(
                feasible,
                key=lambda bs: _objective_value(
                    np.array(bs), mu, Sigma, cfg.lam, 0.0, cfg.k
                ),
            )
            candidate_callback(
                np.array(best),
                {
                    "step": cfg.steps,
                    "batch": batches,
                    "source": source,
                    "unique_feasible_samples": len(seen),
                },
            )
        if len(seen) == comb(n, cfg.k):
            return "exhausted_feasible_space"
    return "time_limit"


def run_qaoa(
    mu: np.ndarray,
    Sigma: np.ndarray,
    cfg: QAOAConfig = QAOAConfig(),
    *,
    candidate_callback=None,
    phase_callback=None,
) -> QAOAResult:
    """Train QAOA with mean cost or lower-tail CVaR, then sample portfolios."""
    set_global_seed(cfg.seed)

    mu, Sigma, n = _validate_inputs(mu, Sigma, cfg)
    H_cost = build_ising_hamiltonian(mu, Sigma, cfg.lam, cfg.alpha, cfg.k)

    dev_train = qml.device(cfg.device, wires=n)

    @qml.qnode(dev_train, interface="autograd")
    def energy(params: np.ndarray):
        _qaoa_ansatz(
            params,
            H_cost=H_cost,
            n=n,
            k=cfg.k,
            mixer=cfg.mixer,
            xy_initial_state=cfg.xy_initial_state,
            xy_topology=cfg.xy_topology,
        )
        return qml.expval(H_cost)

    energy = qml.set_shots(energy, cfg.shots_train)

    def ansatz(params):
        _qaoa_ansatz(
            params, H_cost, n, cfg.k, cfg.mixer, cfg.xy_initial_state, cfg.xy_topology
        )

    objective = make_training_cost(energy, ansatz, dev_train, cfg, mu, Sigma, cfg.lam)

    init = _random_init(cfg.p, cfg.mixer)

    dev_prob = qml.device(cfg.device, wires=n)

    @qml.qnode(dev_prob)
    def state_probs(params: np.ndarray):
        _qaoa_ansatz(
            params,
            H_cost=H_cost,
            n=n,
            k=cfg.k,
            mixer=cfg.mixer,
            xy_initial_state=cfg.xy_initial_state,
            xy_topology=cfg.xy_topology,
        )
        return qml.probs(wires=range(n))

    sample_bits = _sampling_qnode(H_cost, n, cfg)

    def report(step, value, params):
        if candidate_callback is not None:
            report_phase(phase_callback, "candidate_processing")
            checkpoint_probs = np.array(state_probs(params), requires_grad=False)
            counts = _sample_counts_to_counter(sample_bits(params), n)
            feasible = [bs for bs in counts if sum(bs) == cfg.k]
            candidate = (
                min(
                    feasible,
                    key=lambda bs: _objective_value(
                        np.array(bs), mu, Sigma, cfg.lam, 0.0, cfg.k
                    ),
                )
                if feasible
                else _topk_project(
                    _selection_prob_from_probs(checkpoint_probs, n), cfg.k
                )
            )
            candidate_callback(
                np.array(candidate),
                {"step": step, "source": "sample" if feasible else "topk_fallback"},
            )
            report_phase(phase_callback, "optimization")

    report_phase(phase_callback, "optimization")
    opt_res = optimize_training(
        objective,
        init,
        cfg,
        steps=cfg.steps,
        stepsize=cfg.stepsize,
        log_every=cfg.log_every,
        callback=report,
    )

    report_phase(phase_callback, "candidate_processing")
    best_params = np.array(opt_res.params, requires_grad=False)
    gammas = np.array(best_params[:, 0], requires_grad=False)
    betas = np.array(best_params[:, 1], requires_grad=False)

    probs = np.array(state_probs(best_params), requires_grad=False)
    x_prob = _selection_prob_from_probs(probs, n)
    x_round = (x_prob >= 0.5).astype(int)
    x_topk = _topk_project(x_prob, cfg.k)

    samples = sample_bits(best_params)
    counts = _sample_counts_to_counter(samples, n)

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
                alpha=0.0,
                k=cfg.k,
            ),
        )
        x_best_feasible: Optional[np.ndarray] = np.array(best, dtype=int)
    else:
        x_best_feasible = None

    if candidate_callback is not None:
        candidate_callback(
            x_topk if x_best_feasible is None else x_best_feasible,
            {
                "step": cfg.steps,
                "final": True,
                "source": "topk_fallback" if x_best_feasible is None else "sample",
            },
        )
    return QAOAResult(
        params=best_params,
        gammas=gammas,
        betas=betas,
        cost_trace=OptimizeTrace(steps=opt_res.steps, values=opt_res.values),
        state_probs=probs,
        x_prob=np.array(x_prob, requires_grad=False),
        x_round=np.array(x_round, requires_grad=False),
        x_topk=np.array(x_topk, requires_grad=False),
        sample_counts=_ordered_sample_counts(counts),
        x_mode=np.array(x_mode, requires_grad=False),
        x_best_feasible=x_best_feasible,
        **training_metadata(cfg),
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
            _qaoa_ansatz(
                theta,
                H_cost=H_cost,
                n=n,
                k=cfg.k,
                mixer=cfg.mixer,
                xy_initial_state=cfg.xy_initial_state,
                xy_topology=cfg.xy_topology,
            )
            return qml.expval(H_cost)

        energy = qml.set_shots(energy, cfg.shots_train)

        def ansatz(theta):
            _qaoa_ansatz(
                theta,
                H_cost,
                n,
                cfg.k,
                cfg.mixer,
                cfg.xy_initial_state,
                cfg.xy_topology,
            )

        objective = make_training_cost(
            energy, ansatz, dev, cfg, mu, Sigma, float(lam_val)
        )

        if idx == 0 or not sweep.warm_start or params is None:
            init = _random_init(cfg.p, cfg.mixer)
        else:
            init = np.array(params, requires_grad=True)

        opt_res = optimize_training(
            objective,
            init,
            cfg,
            steps=sweep.steps_per_lambda,
            stepsize=sweep.stepsize,
            log_every=max(sweep.steps_per_lambda, 1),
        )
        params = np.array(opt_res.params, requires_grad=False)

        @qml.qnode(dev)
        def state_probs(theta: np.ndarray):
            _qaoa_ansatz(
                theta,
                H_cost=H_cost,
                n=n,
                k=cfg.k,
                mixer=cfg.mixer,
                xy_initial_state=cfg.xy_initial_state,
                xy_topology=cfg.xy_topology,
            )
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
