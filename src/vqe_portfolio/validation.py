"""Small-instance correctness and reliability experiments (not advantage tests)."""

import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import mean

import numpy as np

from .binary import run_binary_vqe
from .classical import (
    convex_markowitz_baseline,
    local_search_binary_baseline,
    mixed_integer_binary_baseline,
)
from .comparison import (
    exact_long_only_markowitz_baseline,
    exhaustive_binary_baseline,
    feasible_sample_rate,
)
from .fractional import run_fractional_vqe
from .problem import PortfolioProblem
from .qaoa import run_qaoa
from .types import BinaryVQEConfig, FractionalVQEConfig, QAOAConfig

METHODS = {
    "exact-binary": ("binary_qubo", False),
    "highs": ("binary_qubo", False),
    "swap": ("binary_qubo", True),
    "random-k": ("binary_qubo", True),
    "binary-vqe": ("binary_qubo", True),
    "qaoa-x": ("binary_qubo", True),
    "qaoa-xy": ("binary_qubo", True),
    "exact-fractional": ("fractional_simplex", False),
    "osqp": ("fractional_simplex", False),
    "fractional-vqe": ("fractional_simplex", True),
}


@dataclass(frozen=True)
class ValidationConfig:
    asset_counts: tuple[int, ...] = (4, 6, 8)
    instance_seeds: tuple[int, ...] = (101, 202, 303)
    solver_seeds: tuple[int, ...] = (0, 1, 2)
    k_fractions: tuple[float, ...] = (0.5,)
    methods: tuple[str, ...] = tuple(METHODS)
    steps: int = 25
    shots: int = 512
    lam: float = 4.0
    alpha: float = 5.0
    binary_depth: int = 2
    qaoa_p: int = 1
    target_gap: float = 1e-3
    solver_time_limit: float = 60.0
    local_restarts: int = 16

    def __post_init__(self):
        for name in ("asset_counts", "instance_seeds", "solver_seeds"):
            values = getattr(self, name)
            if not values or any(type(x) is not int for x in values):
                raise ValueError(f"{name} must contain integers.")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates.")
        if any(n < 2 or n > 16 for n in self.asset_counts):
            raise ValueError("Small-instance validation supports 2–16 assets.")
        if any(s < 0 or s >= 2**32 for s in (*self.instance_seeds, *self.solver_seeds)):
            raise ValueError("Seeds must be in [0, 2**32).")
        if (
            not self.k_fractions
            or any(not 0 < f < 1 for f in self.k_fractions)
            or len(set(self.k_fractions)) != len(self.k_fractions)
        ):
            raise ValueError(
                "Use distinct cardinality fractions strictly inside (0, 1)."
            )
        if not self.methods or set(self.methods) - METHODS.keys():
            raise ValueError("Unknown or empty methods.")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("Methods must not contain duplicates.")
        for name in ("steps", "shots", "binary_depth", "qaoa_p", "local_restarts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.local_restarts < 2:
            raise ValueError("local_restarts must be at least two.")
        for name in ("lam", "alpha", "target_gap", "solver_time_limit"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")

    def to_dict(self):
        # Normalise tuples/lists so a saved JSON configuration compares exactly.
        return {
            k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()
        }


def synthetic_instance(n, seed):
    """Independent factor-model instances with positive definite covariances.

    Volatility and mean scales do not increase with n. This is one instance
    family, not a representative distribution of all real portfolio problems.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, n]))
    mu = rng.uniform(0.04, 0.20, n)
    factors = rng.normal(0, 0.08, (n, min(3, n)))
    sigma = factors @ factors.T + np.diag(rng.uniform(0.01, 0.05, n))
    return mu, sigma


def objective_scale(problem):
    """Fixed instance scale from QUANTUM_ADVANTAGE.md; never divide by f*."""
    k = problem.k if problem.k is not None else 1
    return float(
        abs(problem.lam) * k**2 * np.max(abs(problem.Sigma))
        + k * np.max(abs(problem.mu))
    )


def exact_reference(problem):
    if problem.k is not None:
        return exhaustive_binary_baseline(
            problem.mu, problem.Sigma, problem.lam, 0.0, problem.k
        )
    return exact_long_only_markowitz_baseline(problem.mu, problem.Sigma, problem.lam)


def prepare_instances(config):
    """Freeze snapshots and numerical exact references before solver evaluation."""
    instances = []
    families = {METHODS[m][0] for m in config.methods}
    for n in config.asset_counts:
        for seed in config.instance_seeds:
            mu, sigma = synthetic_instance(n, seed)
            cardinalities = sorted(
                {max(1, min(n - 1, int(n * f))) for f in config.k_fractions}
            )
            tracks = []
            if "binary_qubo" in families:
                tracks.extend(("binary_qubo", k) for k in cardinalities)
            if "fractional_simplex" in families:
                tracks.append(("fractional_simplex", None))
            for family, k in tracks:
                problem = PortfolioProblem(mu, sigma, config.lam, family, k)
                started = time.perf_counter()
                candidate, _ = exact_reference(problem)
                runtime = time.perf_counter() - started
                evaluation = problem.evaluate(candidate)
                if not evaluation.feasible:
                    raise RuntimeError("Exact reference returned an invalid candidate.")
                instances.append(
                    {
                        "problem_id": problem.problem_id,
                        "instance_seed": seed,
                        "snapshot": problem.to_dict(),
                        "reference_candidate": np.asarray(candidate).tolist(),
                        "reference_objective": evaluation.objective,
                        "reference_runtime_seconds": runtime,
                        "objective_scale": objective_scale(problem),
                        "reference_kind": (
                            "exhaustive_binary"
                            if k is not None
                            else "active_set_numerical"
                        ),
                    }
                )
    return instances


def trial_plan(instances, config):
    for instance in instances:
        family = instance["snapshot"]["objective_family"]
        for method in config.methods:
            track, stochastic = METHODS[method]
            if track == family:
                for seed in config.solver_seeds if stochastic else (None,):
                    key = f"{instance['problem_id']}:{method}:{seed}"
                    yield key, instance, method, seed


def solve(problem, method, seed, config):
    """Return a raw candidate and diagnostics, without reference information."""
    mu, sigma, lam, k = problem.mu, problem.Sigma, problem.lam, problem.k
    if method.startswith("exact-"):
        candidate, _ = exact_reference(problem)
        return candidate, {"solver_status": "numerical_optimum"}
    if method == "random-k":
        rng = np.random.default_rng(seed)
        candidates = np.zeros((config.shots, len(mu)), dtype=int)
        for candidate in candidates:
            candidate[rng.choice(len(mu), k, replace=False)] = 1
        scores = (
            lam * np.einsum("bi,ij,bj->b", candidates, sigma, candidates)
            - candidates @ mu
        )
        return candidates[np.argmin(scores)], {
            "solver_status": "completed",
            "sample_feasible_rate": 1.0,
        }
    if method in {"highs", "osqp", "swap"}:
        if method == "highs":
            result = mixed_integer_binary_baseline(
                mu, sigma, lam, k, time_limit=config.solver_time_limit
            )
        elif method == "osqp":
            result = convex_markowitz_baseline(mu, sigma, lam)
        else:
            result = local_search_binary_baseline(
                mu, sigma, lam, k, seed=seed, restarts=config.local_restarts
            )
        return result.candidate, {
            "solver_status": result.status,
            "lower_bound": result.lower_bound,
            "mip_gap": result.mip_gap,
        }
    common = {
        "steps": config.steps,
        "log_every": config.steps,
        "lam": lam,
        "seed": seed,
    }
    if method == "fractional-vqe":
        result = run_fractional_vqe(
            mu, sigma, FractionalVQEConfig(**common, stepsize=0.3)
        )
        return result.weights, {"solver_status": "completed"}
    common.update(k=k, alpha=config.alpha, shots_train=None, shots_sample=config.shots)
    if method == "binary-vqe":
        result = run_binary_vqe(
            mu,
            sigma,
            BinaryVQEConfig(**common, depth=config.binary_depth, stepsize=0.3),
        )
    else:
        result = run_qaoa(
            mu,
            sigma,
            QAOAConfig(
                **common,
                p=config.qaoa_p,
                mixer=method.removeprefix("qaoa-"),
                stepsize=0.2,
            ),
        )
    fallback = result.x_best_feasible is None
    candidate = result.x_topk if fallback else result.x_best_feasible
    return candidate, {
        "solver_status": "completed",
        "fallback_used": fallback,
        "sample_feasible_rate": feasible_sample_rate(result.sample_counts, k),
    }


def evaluate_trial(key, instance, method, seed, config):
    problem = PortfolioProblem.from_dict(instance["snapshot"])
    if problem.problem_id != instance["problem_id"]:
        raise ValueError("Problem snapshot hash mismatch.")
    candidate, diagnostics, error = None, {}, ""
    started = time.perf_counter()
    try:
        candidate, diagnostics = solve(problem, method, seed, config)
    except Exception as exc:  # noqa: BLE001 - failed trials are research outcomes
        # Preserve failures in the denominator; KeyboardInterrupt still stops.
        error = f"{type(exc).__name__}: {exc}"
        diagnostics["solver_status"] = "error"
    runtime = time.perf_counter() - started
    evaluation = problem.evaluate(
        candidate, alpha=config.alpha if problem.k is not None else None
    )
    gap = (
        None
        if not evaluation.feasible
        else evaluation.objective - instance["reference_objective"]
    )
    scale = instance["objective_scale"]
    normalised = gap / scale if gap is not None and scale > 0 else None
    contradiction = normalised is not None and normalised < -1e-8
    # Keep invalid solver output inspectable without emitting nonstandard JSON.
    saved_candidate = None
    if candidate is not None:
        saved_candidate = (
            np.asarray(candidate).tolist() if evaluation.feasible else repr(candidate)
        )
    for field in ("lower_bound", "mip_gap"):
        value = diagnostics.get(field)
        if value is not None and not np.isfinite(value):
            diagnostics[field] = None
            diagnostics[f"nonfinite_{field}"] = repr(value)
    return {
        "trial_id": key,
        "problem_id": problem.problem_id,
        "instance_seed": instance["instance_seed"],
        "n_assets": len(problem.mu),
        "k": problem.k,
        "objective_family": problem.objective_family,
        "method": method,
        "seed": seed,
        "runtime_seconds": runtime,
        "candidate": saved_candidate,
        "feasible": bool(evaluation.feasible),
        "objective": evaluation.objective,
        "reference_objective": instance["reference_objective"],
        "objective_scale": scale,
        "absolute_gap": gap,
        "normalised_gap": normalised,
        "target_hit": bool(
            normalised is not None
            and not contradiction
            and normalised <= config.target_gap
        ),
        "reference_contradiction": contradiction,
        "validation_error": evaluation.reason,
        "error": error,
        **diagnostics,
    }


def summarise_trials(trials):
    """Equal-weight instance means; failure rates always include every trial.

    Quality is conditional on feasible returns. No uncertainty estimates or
    advantage decisions are made by this pilot summary.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for row in trials:
        group = (row["n_assets"], row["k"], row["objective_family"], row["method"])
        groups[group][row["problem_id"]].append(row)
    summaries = []
    for (n, k, family, method), instances in groups.items():
        all_rows = [row for rows in instances.values() for row in rows]
        instance_gaps = [
            mean(r["normalised_gap"] for r in rows if r["normalised_gap"] is not None)
            for rows in instances.values()
            if any(r["normalised_gap"] is not None for r in rows)
        ]
        summaries.append(
            {
                "n_assets": n,
                "k": k,
                "objective_family": family,
                "method": method,
                "instance_count": len(instances),
                "trial_count": len(all_rows),
                "quality_instance_count": len(instance_gaps),
                "feasible_rate": mean(
                    mean(r["feasible"] for r in rows) for rows in instances.values()
                ),
                "target_hit_rate": mean(
                    mean(r["target_hit"] for r in rows) for rows in instances.values()
                ),
                "mean_normalised_gap_feasible": (
                    mean(instance_gaps) if instance_gaps else None
                ),
                "mean_runtime_seconds": mean(
                    mean(r["runtime_seconds"] for r in rows)
                    for rows in instances.values()
                ),
                "error_count": sum(bool(r["error"]) for r in all_rows),
                "reference_contradictions": sum(
                    r["reference_contradiction"] for r in all_rows
                ),
            }
        )
    return summaries
