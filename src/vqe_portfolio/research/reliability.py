"""Controlled development comparisons of opt-in solver reliability controls."""

import time
from dataclasses import asdict

import numpy as np
import pennylane as qml

from ..ansatz import apply_binary_ansatz
from ..binary import build_ising_hamiltonian, run_binary_vqe
from ..problem import PortfolioProblem
from ..qaoa import _qaoa_ansatz, run_qaoa
from ..types import BinaryVQEConfig, QAOAConfig
from .training_diagnostics import distribution_metrics, target_basis


VARIANTS = (
    "binary-legacy",
    "binary-auto",
    "binary-auto-normalized",
    "binary-restarts",
    "xy-basis-ring",
    "xy-dicke-ring",
    "xy-basis-complete",
    "xy-dicke-complete",
)


def variant_config(variant, problem, seed, steps, shots, restarts=3):
    if variant not in VARIANTS:
        raise ValueError("Unknown reliability variant")
    common = dict(
        k=problem.k,
        lam=problem.lam,
        seed=seed,
        steps=steps,
        log_every=steps,
        shots_sample=shots,
        alpha=5.0,
    )
    if variant.startswith("xy-"):
        _, initial, topology = variant.split("-")
        return QAOAConfig(
            **common,
            p=1,
            stepsize=0.2,
            mixer="xy",
            xy_initial_state=initial,
            xy_topology=topology,
        )
    if variant != "binary-legacy":
        common["alpha"] = "auto"
    if variant == "binary-restarts":
        if steps < restarts or steps % restarts:
            raise ValueError("Total steps must be a positive multiple of restarts")
        common["steps"] = steps // restarts
        common["log_every"] = common["steps"]
    return BinaryVQEConfig(
        **common,
        normalize_energy=variant in {"binary-auto-normalized", "binary-restarts"},
        restarts=restarts if variant == "binary-restarts" else 1,
        keep_best=variant == "binary-restarts",
        stepsize_decay=0.03 if variant == "binary-restarts" else 0.0,
    )


def measure_variant(
    instance, variant, seed, steps, shots=64, restarts=3, target_gap=1e-3
):
    problem = PortfolioProblem.from_dict(instance["snapshot"])
    if problem.problem_id != instance["problem_id"]:
        raise ValueError("Problem snapshot hash mismatch")
    cfg = variant_config(variant, problem, seed, steps, shots, restarts)
    row = {
        "trial_id": f"{problem.problem_id}:{variant}:{seed}",
        "problem_id": problem.problem_id,
        "instance_seed": instance["instance_seed"],
        "variant": variant,
        "seed": seed,
        "n_assets": len(problem.mu),
        "k": problem.k,
        "config": asdict(cfg),
        "total_training_steps": steps,
        "error": "",
    }
    started = time.perf_counter()
    try:
        if variant.startswith("binary-"):
            result = run_binary_vqe(problem.mu, problem.Sigma, cfg)
            alpha = result.effective_alpha
        else:
            result = run_qaoa(problem.mu, problem.Sigma, cfg)
            alpha = cfg.alpha
        row["solver_seconds"] = time.perf_counter() - started
        params = result.params
        h = build_ising_hamiltonian(
            problem.mu, problem.Sigma, problem.lam, alpha, problem.k
        )

        @qml.qnode(qml.device(cfg.device, wires=len(problem.mu)))
        def probabilities():
            if variant.startswith("binary-"):
                apply_binary_ansatz(cfg.ansatz, params, cfg.depth, len(problem.mu))
            else:
                _qaoa_ansatz(
                    params,
                    h,
                    len(problem.mu),
                    cfg.k,
                    cfg.mixer,
                    cfg.xy_initial_state,
                    cfg.xy_topology,
                )
            return qml.probs(wires=range(len(problem.mu)))

        probs = np.asarray(probabilities(), dtype=float)
        bits, feasible, target, objectives = target_basis(
            problem, instance["reference_objective"], target_gap
        )
        energy = objectives + alpha * (bits.sum(axis=1) - problem.k) ** 2
        row.update(distribution_metrics(probs, feasible, target, energy))
        uniform = float(target.sum() / feasible.sum())
        row["uniform_success_64"] = (
            1.0 if uniform == 1 else float(-np.expm1(64 * np.log1p(-uniform)))
        )
        row.update(
            params=np.asarray(params).tolist(),
            probabilities=probs.tolist(),
            effective_alpha=float(alpha),
        )
        for draws in (16, 64, 512):
            row[f"success_{draws}"] = (
                float(-np.expm1(draws * np.log1p(-min(row["p_target"], 1.0))))
                if row["p_target"] < 1
                else 1.0
            )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["solver_seconds"] = time.perf_counter() - started
    return row
