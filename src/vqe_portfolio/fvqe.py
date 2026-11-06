"""Filtering VQE for diagonal portfolio Hamiltonians.

Implements Eqs. (6)–(7) of Amaro et al., arXiv:2106.10055, rather than
minimizing the expectation of a transformed Hamiltonian. Each iteration takes
one gradient step toward the freshly filtered current state. Computational-basis
measurements suffice; no filtered state or dense operator is constructed.
"""

from collections import Counter
from numbers import Real

import numpy as np
import pennylane as qml

from .ansatz import apply_binary_ansatz, binary_ansatz_shape
from .binary import _ordered_sample_counts, topk_project
from .cvar import _portfolio_costs
from .optimize import report_phase
from .problem import PortfolioProblem
from .reliability import resolve_penalty
from .types import FVQEConfig, FVQEResult, OptimizeTrace


def _validate_config(cfg):
    for name, minimum in (
        ("depth", 1),
        ("steps", 0),
        ("log_every", 1),
        ("shots_sample", 1),
        ("tau_search_steps", 1),
    ):
        value = getattr(cfg, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if cfg.shots_train is not None and (
        type(cfg.shots_train) is not int or cfg.shots_train < 1
    ):
        raise ValueError("shots_train must be a positive integer or None")
    for name in ("stepsize", "tau", "tau_max", "gradient_target", "spectrum_floor"):
        value = getattr(cfg, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if cfg.spectrum_floor >= 1:
        raise ValueError("spectrum_floor must be in (0, 1)")
    if cfg.tau_max < cfg.tau:
        raise ValueError("tau_max must be >= tau")
    if type(cfg.adapt_tau) is not bool:
        raise ValueError("adapt_tau must be a boolean")
    if cfg.filter_type not in {"inverse", "exponential"}:
        raise ValueError("filter_type must be 'inverse' or 'exponential'")
    if type(cfg.seed) is not int or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def _spectrum_bounds(mu, sigma, lam, alpha, k):
    """Coefficient bounds, without finding any extremal portfolio or eigenvalue."""
    linear = lam * np.diag(sigma) - mu + alpha * (1 - 2 * k)
    pairs = 2 * lam * sigma[np.triu_indices(len(mu), 1)] + 2 * alpha
    constant = alpha * k**2
    lower = float(constant + np.minimum(linear, 0).sum() + np.minimum(pairs, 0).sum())
    upper = float(constant + np.maximum(linear, 0).sum() + np.maximum(pairs, 0).sum())
    if not np.all(np.isfinite([lower, upper, upper - lower])):
        raise ValueError("Portfolio coefficients exceed the finite energy range")
    return lower, upper


def _scaled_energies(energies, lower, upper, floor):
    """A positive affine spectrum for inverse filtering, shared by both filters."""
    if upper == lower:
        return np.ones_like(energies, dtype=float)
    unit = np.clip((energies - lower) / (upper - lower), 0.0, 1.0)
    return floor + (1.0 - floor) * unit


def _log_filter_mean(distribution, filter_type, tau, power=1):
    energies, probabilities = distribution
    mask = probabilities > 0
    log_filter = (
        -np.log(energies[mask]) if filter_type == "inverse" else -energies[mask]
    )
    return np.logaddexp.reduce(np.log(probabilities[mask]) + (power * tau) * log_filter)


def _filter_gradient(center, plus, minus, filter_type, tau):
    """Frozen-target distance gradient, with one common sqrt(<F²>) denominator.

    Inputs contain normalized energies and probability mass (or empirical mass).
    Log-space moments avoid overflow/underflow of the unnormalized filter.
    """
    if tau == 0:
        return np.zeros(len(plus))
    log_norm = 0.5 * _log_filter_mean(center, filter_type, tau, power=2)
    with np.errstate(over="ignore", invalid="ignore"):
        gradient = np.array(
            [
                -0.25
                * (
                    np.exp(_log_filter_mean(p, filter_type, tau) - log_norm)
                    - np.exp(_log_filter_mean(m, filter_type, tau) - log_norm)
                )
                for p, m in zip(plus, minus)
            ]
        )
    if not np.all(np.isfinite(gradient)):
        raise ValueError("Filter gradient overflow; reduce tau or tau_max")
    return gradient


def _choose_tau(gradient_at, cfg):
    """Bounded version of Sec. II.4: bracket, then bisect below the threshold.

    Every call uses the same cached circuit outcomes. If the target cannot be
    bracketed by tau_max, choose the closest norm below it among tried values.
    This is filter selection, not rescaling a gradient to a prescribed norm.
    """
    if not cfg.adapt_tau:
        return cfg.tau, gradient_at(cfg.tau)
    best_tau, best_gradient, best_norm = 0.0, None, -1.0

    def evaluate(tau):
        nonlocal best_tau, best_gradient, best_norm
        gradient = gradient_at(tau)
        norm = float(np.linalg.norm(gradient))
        if norm <= cfg.gradient_target and norm > best_norm:
            best_tau, best_gradient, best_norm = tau, gradient, norm
        return norm

    upper = cfg.tau
    while evaluate(upper) <= cfg.gradient_target:
        if upper >= cfg.tau_max:
            return best_tau, best_gradient
        upper = min(2.0 * upper, cfg.tau_max)
    lower = 0.0
    # At tau=0 the exact and empirical gradients vanish.
    evaluate(lower)
    for _ in range(cfg.tau_search_steps):
        midpoint = (lower + upper) / 2.0
        if evaluate(midpoint) <= cfg.gradient_target:
            lower = midpoint
        else:
            upper = midpoint
    return best_tau, best_gradient


def run_fvqe(
    mu,
    Sigma,
    cfg: FVQEConfig = FVQEConfig(),
    *,
    candidate_callback=None,
    phase_callback=None,
) -> FVQEResult:
    """Train F-VQE and return a measured exact-K candidate (or Top-K fallback).

    Analytic execution uses basis probabilities. Finite-shot execution only
    scores measured bitstrings and never enumerates the spectrum. Each update
    uses 2m shifted circuits plus the current circuit for m parameters; the
    final state costs one more training circuit. Adaptive tau uses no extra
    circuits. Callbacks at step 0 and logging checkpoints also sample candidates.
    """
    _validate_config(cfg)
    problem = PortfolioProblem(mu, Sigma, cfg.lam, "binary_qubo", cfg.k)
    mu, sigma, n = problem.mu, problem.Sigma, len(problem.mu)
    if cfg.k < 1:
        raise ValueError(f"k must be in [1, {n}]")
    shape = binary_ansatz_shape(cfg.ansatz, cfg.depth, n)
    alpha = resolve_penalty(mu, sigma, cfg.lam, cfg.k, cfg.alpha)
    lower, upper = _spectrum_bounds(mu, sigma, cfg.lam, alpha, cfg.k)
    rng = np.random.default_rng(cfg.seed)
    params = rng.uniform(0, np.pi, size=shape)
    # Local RNGs isolate training from checkpoint/final sampling and other solvers.
    dev = qml.device(cfg.device, wires=n, seed=cfg.seed)
    sample_dev = qml.device(cfg.device, wires=n, seed=cfg.seed + 1)

    def ansatz(theta):
        apply_binary_ansatz(cfg.ansatz, theta, cfg.depth, n)

    if cfg.shots_train is None:
        indices = np.arange(2**n)
        bits = (indices[:, None] >> np.arange(n - 1, -1, -1)) & 1
        energies = _portfolio_costs(bits, mu, sigma, cfg.lam, alpha, cfg.k)
        scaled = _scaled_energies(energies, lower, upper, cfg.spectrum_floor)

        @qml.qnode(dev)
        def training_circuit(theta):
            ansatz(theta)
            return qml.probs(wires=range(n))

        def distribution(theta):
            probabilities = np.maximum(np.asarray(training_circuit(theta)), 0)
            probabilities = probabilities / probabilities.sum()
            return (scaled, probabilities), float(probabilities @ energies)

    else:

        @qml.qnode(dev)
        def training_circuit(theta):
            ansatz(theta)
            return qml.sample(wires=range(n))

        training_circuit = qml.set_shots(training_circuit, cfg.shots_train)

        def distribution(theta):
            rows = np.asarray(training_circuit(theta)).reshape(cfg.shots_train, n)
            costs = _portfolio_costs(rows, mu, sigma, cfg.lam, alpha, cfg.k)
            scaled = _scaled_energies(costs, lower, upper, cfg.spectrum_floor)
            return (scaled, np.full(len(costs), 1.0 / len(costs))), float(costs.mean())

    @qml.qnode(sample_dev)
    def sample_circuit(theta):
        ansatz(theta)
        return qml.sample(wires=range(n))

    sample_circuit = qml.set_shots(sample_circuit, cfg.shots_sample)

    def measure(theta):
        rows = np.asarray(sample_circuit(theta)).reshape(cfg.shots_sample, n)
        counts = Counter(tuple(map(int, row)) for row in rows)
        if cfg.shots_train is None:
            probabilities = center[1]
            x_prob = probabilities @ bits
        else:
            # Marginals and candidate counts share a single final sampling batch.
            x_prob = rows.mean(axis=0)
        topk = topk_project(x_prob, cfg.k)
        feasible = [bs for bs in counts if sum(bs) == cfg.k]
        best = (
            np.array(
                min(feasible, key=lambda bs: (problem.evaluate(bs).objective, bs)),
                dtype=int,
            )
            if feasible
            else None
        )
        return x_prob, topk, counts, best

    def publish(step, measured, final=False):
        if candidate_callback is not None:
            _, topk, _, best = measured
            candidate_callback(
                (topk if best is None else best).copy(),
                {
                    "step": step,
                    "source": "topk_fallback" if best is None else "sample",
                    "final": final,
                },
            )

    report_phase(phase_callback, "optimization")
    center, energy = distribution(params)
    energy_steps, energy_values = [0], [energy]
    taus, norms = [], []
    if candidate_callback is not None and cfg.steps:
        report_phase(phase_callback, "candidate_processing")
        publish(0, measure(params))
        report_phase(phase_callback, "optimization")
    for step in range(1, cfg.steps + 1):
        plus, minus = [], []
        for index in np.ndindex(shape):
            shifted = params.copy()
            shifted[index] += np.pi / 2
            plus.append(distribution(shifted)[0])
            shifted[index] -= np.pi
            minus.append(distribution(shifted)[0])
        tau, gradient = _choose_tau(
            lambda strength: _filter_gradient(
                center, plus, minus, cfg.filter_type, strength
            ),
            cfg,
        )
        params = params - cfg.stepsize * gradient.reshape(shape)
        if not np.all(np.isfinite(params)):
            raise ValueError("Nonfinite F-VQE update; reduce stepsize or tau")
        taus.append(float(tau))
        norms.append(float(np.linalg.norm(gradient)))
        center, energy = distribution(params)
        if step % cfg.log_every == 0 or step == cfg.steps:
            energy_steps.append(step)
            energy_values.append(energy)
            if candidate_callback is not None and step != cfg.steps:
                report_phase(phase_callback, "candidate_processing")
                publish(step, measure(params))
                report_phase(phase_callback, "optimization")

    report_phase(phase_callback, "candidate_processing")
    measured = measure(params)
    publish(cfg.steps, measured, final=True)
    x_prob, topk, counts, best = measured
    return FVQEResult(
        params=params,
        energy_trace=OptimizeTrace(energy_steps, energy_values),
        tau_trace=OptimizeTrace(list(range(1, cfg.steps + 1)), taus),
        gradient_norm_trace=OptimizeTrace(list(range(1, cfg.steps + 1)), norms),
        z_expect=1.0 - 2.0 * x_prob,
        x_prob=x_prob,
        x_round=(x_prob >= 0.5).astype(int),
        x_topk=topk,
        sample_counts=_ordered_sample_counts(counts),
        x_mode=np.array(max(counts, key=counts.get), dtype=int),
        x_best_feasible=best,
        effective_alpha=alpha,
        filter_type=cfg.filter_type,
        adapt_tau=cfg.adapt_tau,
        spectrum_lower_bound=lower,
        spectrum_upper_bound=upper,
        spectrum_floor=cfg.spectrum_floor,
        training_circuit_evaluations=cfg.steps * (2 * params.size + 1) + 1,
    )
