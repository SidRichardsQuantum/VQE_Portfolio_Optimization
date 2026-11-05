"""Analytical iid sampling allowances using frozen circuit distributions."""

from collections import defaultdict
from numbers import Integral
from statistics import mean

import numpy as np

from .problem import PortfolioProblem
from .training_diagnostics import distribution_metrics, target_basis
from .validation import ValidationConfig, trial_plan


def validate_draw_counts(draw_counts):
    if not draw_counts or any(
        isinstance(m, bool) or not isinstance(m, Integral) or m < 1 for m in draw_counts
    ):
        raise ValueError("Draw allowances must be positive integers.")
    if len(set(draw_counts)) != len(draw_counts):
        raise ValueError("Draw allowances must be distinct.")
    return tuple(sorted(int(m) for m in draw_counts))


def success_probability(p, draws):
    """P(at least one target in m independent draws), stable even for tiny p."""
    validate_draw_counts((draws,))
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Single-draw probability must be finite and in [0, 1].")
    if p == 0 or p == 1:
        return float(p)
    return float(-np.expm1(draws * np.log1p(-p)))


def sampling_rows(manifest, records, draw_counts=(16, 64, 512)):
    """Validate complete source evidence; transform per pair, then aggregate.

    Existing failed diagnostic pairs remain present with blank probabilities.
    Corrupted distributions/metadata and missing or duplicate pairs fail loudly.
    The classical baseline appears once per problem and allowance, not once per
    quantum method/seed. Nothing is trained or sampled here.
    """
    draws = validate_draw_counts(draw_counts)
    config = ValidationConfig(**manifest["config"])
    instances = manifest["instances"]
    if not instances or len({i["problem_id"] for i in instances}) != len(instances):
        raise ValueError("Require distinct source instances.")
    plan = {
        key: (instance, method, seed)
        for key, instance, method, seed in trial_plan(instances, config)
    }
    ids = [r["trial_id"] for r in records]
    if len(set(ids)) != len(ids) or set(ids) != set(plan):
        raise ValueError(
            "Require a complete diagnostic checkpoint with distinct planned pairs."
        )
    prepared, baselines = {}, []
    for instance in instances:
        problem = PortfolioProblem.from_dict(instance["snapshot"])
        if problem.problem_id != instance["problem_id"]:
            raise ValueError("Problem snapshot hash mismatch.")
        bits, feasible, target, objectives = target_basis(
            problem, instance["reference_objective"], config.target_gap
        )
        energies = objectives + config.alpha * (bits.sum(axis=1) - problem.k) ** 2
        uniform_p = float(target.sum() / feasible.sum())
        prepared[problem.problem_id] = (feasible, target, energies, uniform_p)
        for m in draws:
            baselines.append(
                {
                    "problem_id": problem.problem_id,
                    "instance_seed": instance["instance_seed"],
                    "n_assets": len(problem.mu),
                    "k": problem.k,
                    "draws": m,
                    "target_count": int(target.sum()),
                    "feasible_count": int(feasible.sum()),
                    "p_target": uniform_p,
                    "success_probability": success_probability(uniform_p, m),
                }
            )
    pairs = []
    for record in records:
        instance, method, seed = plan[record["trial_id"]]
        for field, expected in (
            ("problem_id", instance["problem_id"]),
            ("method", method),
            ("seed", seed),
            ("instance_seed", instance["instance_seed"]),
            ("n_assets", len(instance["snapshot"]["mu"])),
            ("k", instance["snapshot"]["k"]),
        ):
            if record[field] != expected:
                raise ValueError(f"Source pair metadata mismatch: {field}.")
        feasible, target, energies, uniform_p = prepared[record["problem_id"]]
        probabilities = {}
        if not record["error"]:
            if record["target_basis_indices"] != np.flatnonzero(target).tolist():
                raise ValueError(
                    "Source target indices disagree with the frozen objective."
                )
            for phase in ("before", "after"):
                metrics = distribution_metrics(
                    record[f"{phase}_probabilities"], feasible, target, energies
                )
                if not np.isclose(
                    metrics["p_target"],
                    record[f"{phase}_p_target"],
                    rtol=1e-10,
                    atol=1e-15,
                ):
                    raise ValueError(
                        "Stored target probability disagrees with raw distribution."
                    )
                probabilities[phase] = metrics["p_target"]
        for m in draws:
            initial = (
                success_probability(probabilities["before"], m)
                if probabilities
                else None
            )
            trained = (
                success_probability(probabilities["after"], m)
                if probabilities
                else None
            )
            pairs.append(
                {
                    "source_trial_id": record["trial_id"],
                    "problem_id": record["problem_id"],
                    "instance_seed": record["instance_seed"],
                    "n_assets": record["n_assets"],
                    "k": record["k"],
                    "method": method,
                    "seed": seed,
                    "draws": m,
                    "initial_p_target": probabilities.get("before"),
                    "trained_p_target": probabilities.get("after"),
                    "initial_success_probability": initial,
                    "trained_success_probability": trained,
                    "training_change": trained - initial if probabilities else None,
                    "uniform_feasible_success_probability": success_probability(
                        uniform_p, m
                    ),
                    "error": record["error"],
                }
            )
    return pairs, baselines


def summarise_sampling(pairs, baselines):
    groups = defaultdict(lambda: defaultdict(list))
    for row in pairs:
        groups[(row["n_assets"], row["k"], row["method"], row["draws"])][
            row["problem_id"]
        ].append(row)
    summaries = []
    for (n, k, method, draws), instances in groups.items():
        all_rows = [r for rows in instances.values() for r in rows]
        valid = [[r for r in rows if not r["error"]] for rows in instances.values()]
        valid = [rows for rows in valid if rows]
        summary = {
            "n_assets": n,
            "k": k,
            "method": method,
            "draws": draws,
            "instance_count": len(instances),
            "pair_count": len(all_rows),
            "successful_pairs": sum(len(rows) for rows in valid),
            "quality_instance_count": len(valid),
            "error_count": sum(bool(r["error"]) for r in all_rows),
        }
        for field in (
            "initial_success_probability",
            "trained_success_probability",
            "training_change",
            "uniform_feasible_success_probability",
        ):
            summary[f"mean_{field}"] = (
                mean(mean(r[field] for r in rows) for rows in valid) if valid else None
            )
        summaries.append(summary)
    baseline_groups = defaultdict(list)
    for row in baselines:
        baseline_groups[(row["n_assets"], row["k"], row["draws"])].append(row)
    classical = [
        {
            "n_assets": n,
            "k": k,
            "draws": draws,
            "instance_count": len(rows),
            "mean_success_probability": mean(r["success_probability"] for r in rows),
        }
        for (n, k, draws), rows in baseline_groups.items()
    ]
    return summaries, classical
