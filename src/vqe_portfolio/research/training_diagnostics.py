"""Paired exact-distribution diagnostics for binary portfolio circuits."""

import time
from collections import defaultdict
from dataclasses import asdict
from math import gcd
from numbers import Integral
from statistics import mean

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ..ansatz import apply_binary_ansatz, binary_ansatz_shape
from ..binary import build_ising_hamiltonian
from ..optimize import adam_optimize
from ..problem import PortfolioProblem
from ..qaoa import _qaoa_ansatz, _random_init
from ..types import BinaryVQEConfig, QAOAConfig
from ..utils import set_global_seed
from .validation import objective_scale

DIAGNOSTIC_METHODS = ("binary-vqe", "qaoa-x", "qaoa-xy")


def target_basis(problem, reference_objective, target_gap):
    """Enumerate in PennyLane wire order, testing the original objective."""
    n = len(problem.mu)
    if problem.k is None or not 2 <= n <= 16:
        raise ValueError("Diagnostic requires a binary problem with 2–16 assets.")
    scale = objective_scale(problem)
    if not np.isfinite(reference_objective) or scale <= 0:
        raise ValueError("Require a finite reference and positive objective scale.")
    if not np.isfinite(target_gap) or target_gap < 0:
        raise ValueError("target_gap must be finite and nonnegative.")
    bits = (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
    counts = bits.sum(axis=1)
    feasible = counts == problem.k
    objectives = (
        problem.lam * np.einsum("bi,ij,bj->b", bits, problem.Sigma, bits)
        - bits @ problem.mu
    )
    optimum = float(objectives[feasible].min())
    if abs(optimum - reference_objective) > 1e-8 * scale:
        raise ValueError("Frozen reference disagrees with exhaustive enumeration.")
    target = feasible & ((objectives - reference_objective) / scale <= target_gap)
    return bits, feasible, target, objectives


def distribution_metrics(probabilities, feasible, target, energies):
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        probabilities.shape != feasible.shape
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or abs(probabilities.sum() - 1) > 1e-8
    ):
        raise ValueError("Invalid exact probability distribution.")
    feasible_mass = float(probabilities[feasible].sum())
    target_mass = float(probabilities[target].sum())
    return {
        "probability_sum": float(probabilities.sum()),
        "p_feasible": min(1.0, feasible_mass),
        "p_target": min(1.0, target_mass),
        "p_target_given_feasible": (
            target_mass / feasible_mass if feasible_mass > 0 else None
        ),
        "expected_penalized_energy": float(probabilities @ energies),
    }


def validate_checkpoints(steps, final_step):
    if (
        len(steps) < 2
        or any(
            isinstance(s, bool) or not isinstance(s, Integral) or s < 0 for s in steps
        )
        or len(set(steps)) != len(steps)
        or min(steps) != 0
        or max(steps) != final_step
    ):
        raise ValueError(
            "Checkpoints must be distinct nonnegative integers including zero and the final step."
        )
    return tuple(sorted(int(s) for s in steps))


def train_checkpoints(problem, method, seed, config, checkpoint_steps):
    """One uninterrupted production Adam trajectory, without references.

    The private QAOA helpers are deliberately shared with the production path.
    Regression tests compare final parameters/probabilities with public solvers.
    Parameter snapshots and cumulative optimisation times are captured in the
    callback; all distribution evaluations happen after optimisation finishes.
    """
    checkpoint_steps = validate_checkpoints(checkpoint_steps, config.steps)
    if method not in DIAGNOSTIC_METHODS or problem.k is None:
        raise ValueError("Use a supported binary circuit method.")
    common = {
        "lam": problem.lam,
        "alpha": config.alpha,
        "k": problem.k,
        "seed": seed,
        "steps": config.steps,
        "log_every": gcd(*checkpoint_steps[1:]),
        "shots_train": None,
        "shots_sample": config.shots,
    }
    if method == "binary-vqe":
        cfg = BinaryVQEConfig(**common, depth=config.binary_depth, stepsize=0.3)
    else:
        cfg = QAOAConfig(
            **common, p=config.qaoa_p, mixer=method.removeprefix("qaoa-"), stepsize=0.2
        )
    set_global_seed(seed)
    n = len(problem.mu)
    hamiltonian = build_ising_hamiltonian(
        problem.mu, problem.Sigma, cfg.lam, cfg.alpha, cfg.k
    )
    device = qml.device(cfg.device, wires=n)

    def ansatz(params):
        if method == "binary-vqe":
            apply_binary_ansatz(cfg.ansatz, params, depth=cfg.depth, n_wires=n)
        else:
            _qaoa_ansatz(params, H_cost=hamiltonian, n=n, k=cfg.k, mixer=cfg.mixer)

    @qml.qnode(device, interface="autograd")
    def energy(params):
        ansatz(params)
        return qml.expval(hamiltonian)

    if method == "binary-vqe":
        shape = binary_ansatz_shape(cfg.ansatz, cfg.depth, n)
        init = pnp.array(pnp.random.uniform(0, pnp.pi, size=shape), requires_grad=True)
    else:
        init = _random_init(cfg.p, cfg.mixer)
    initial = np.asarray(init).copy()
    snapshots = [(0, initial, 0.0)]
    started = time.perf_counter()

    def capture(step, value, params):
        if step in checkpoint_steps:
            elapsed = time.perf_counter() - started
            snapshots.append((step, np.asarray(params).copy(), elapsed))

    adam_optimize(
        energy,
        init,
        steps=cfg.steps,
        stepsize=cfg.stepsize,
        log_every=cfg.log_every,
        callback=capture,
    )
    if tuple(s[0] for s in snapshots) != checkpoint_steps:
        raise RuntimeError("Optimisation did not capture every requested checkpoint.")

    @qml.qnode(device)
    def probabilities(params):
        ansatz(params)
        return qml.probs(wires=range(n))

    return {
        "algorithm_config": asdict(cfg),
        "checkpoints": [
            {
                "step": step,
                "params": params.tolist(),
                "probabilities": np.asarray(
                    probabilities(params), dtype=float
                ).tolist(),
                "training_runtime_seconds": elapsed,
            }
            for step, params, elapsed in snapshots
        ],
    }


def train_circuit(problem, method, seed, config):
    """Legacy initial/final diagnostic, sharing the checkpoint training path."""
    result = train_checkpoints(problem, method, seed, config, (0, config.steps))
    initial, trained = result["checkpoints"]
    return {
        "algorithm_config": result["algorithm_config"],
        "initial_params": initial["params"],
        "trained_params": trained["params"],
        "before_probabilities": initial["probabilities"],
        "after_probabilities": trained["probabilities"],
        "training_runtime_seconds": trained["training_runtime_seconds"],
    }


def measure_pair(key, instance, method, seed, config):
    row = {
        "trial_id": key,
        "problem_id": instance["problem_id"],
        "instance_seed": instance["instance_seed"],
        "method": method,
        "seed": seed,
        "n_assets": len(instance["snapshot"]["mu"]),
        "k": instance["snapshot"]["k"],
        "error": "",
    }
    try:
        problem = PortfolioProblem.from_dict(instance["snapshot"])
        if problem.problem_id != instance["problem_id"]:
            raise ValueError("Problem snapshot hash mismatch.")
        bits, feasible, target, objectives = target_basis(
            problem, instance["reference_objective"], config.target_gap
        )
        energies = objectives + config.alpha * (bits.sum(axis=1) - problem.k) ** 2
        result = train_circuit(problem, method, seed, config)
        before = distribution_metrics(
            result["before_probabilities"], feasible, target, energies
        )
        after = distribution_metrics(
            result["after_probabilities"], feasible, target, energies
        )
        row.update(result)
        row.update({f"before_{name}": value for name, value in before.items()})
        row.update({f"after_{name}": value for name, value in after.items()})
        row.update(
            {
                "delta_p_target": after["p_target"] - before["p_target"],
                "delta_p_feasible": after["p_feasible"] - before["p_feasible"],
                "target_basis_indices": np.flatnonzero(target).tolist(),
                "feasible_count": int(feasible.sum()),
                "target_count": int(target.sum()),
                "uniform_bitstring_p_target": float(target.mean()),
                "uniform_feasible_p_target": float(target.sum() / feasible.sum()),
            }
        )
    except Exception as exc:  # noqa: BLE001 - retain failed diagnostic pairs
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def summarise_pairs(trials):
    groups = defaultdict(lambda: defaultdict(list))
    for row in trials:
        groups[(row["n_assets"], row["k"], row["method"])][row["problem_id"]].append(
            row
        )
    summaries = []
    for (n, k, method), instances in groups.items():
        all_rows = [row for rows in instances.values() for row in rows]
        successful = [
            [r for r in rows if not r["error"]] for rows in instances.values()
        ]
        successful = [rows for rows in successful if rows]
        summary = {
            "n_assets": n,
            "k": k,
            "method": method,
            "instance_count": len(instances),
            "pair_count": len(all_rows),
            "successful_pairs": sum(len(rows) for rows in successful),
            "quality_instance_count": len(successful),
            "error_count": sum(bool(r["error"]) for r in all_rows),
            "improved_pairs": sum(r.get("delta_p_target", 0) > 1e-12 for r in all_rows),
            "worsened_pairs": sum(
                r.get("delta_p_target", 0) < -1e-12 for r in all_rows
            ),
        }
        for field in (
            "before_p_target",
            "after_p_target",
            "delta_p_target",
            "before_p_feasible",
            "after_p_feasible",
            "before_p_target_given_feasible",
            "after_p_target_given_feasible",
            "uniform_feasible_p_target",
        ):
            values = [
                mean(r[field] for r in rows if r[field] is not None)
                for rows in successful
                if any(r[field] is not None for r in rows)
            ]
            summary[f"mean_{field}"] = mean(values) if values else None
        summaries.append(summary)
    return summaries
