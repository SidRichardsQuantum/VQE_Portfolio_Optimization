"""Fixed-deadline trial evaluation; references stay outside solver execution."""

import math
from collections import defaultdict
from dataclasses import asdict
from statistics import mean

from ..execution import run_budgeted
from ..problem import PortfolioProblem


BUDGET_VARIANTS = {
    "binary-legacy": "binary-vqe",
    "binary-improved": "binary-vqe",
    "xy-basis-ring": "qaoa-xy",
    "xy-dicke-ring-trained": "qaoa-xy-trained",
    "xy-dicke-ring-untrained": "qaoa-xy-untrained",
}


def evaluate_budget_trial(
    instance, method, seed, seconds, config=None, target_gap=1e-3, *, variant=None
):
    problem = PortfolioProblem.from_dict(instance["snapshot"])
    if problem.problem_id != instance["problem_id"]:
        raise ValueError("Problem snapshot hash mismatch")
    result = run_budgeted(problem, method, seconds, config=config, seed=seed)
    row = asdict(result)
    scale = instance["objective_scale"]
    reference = instance["reference_objective"]
    gap = None if result.objective is None else result.objective - reference
    normalized = gap / scale if gap is not None and scale > 0 else None
    target_times = [
        s.elapsed_seconds
        for s in result.incumbents
        if scale > 0 and -1e-8 <= (s.objective - reference) / scale <= target_gap
    ]
    return {
        **row,
        "trial_id": f"{problem.problem_id}:{variant or method}:{seed}:{seconds:g}",
        "variant": variant or method,
        "seed": seed,
        "instance_seed": instance["instance_seed"],
        "n_assets": len(problem.mu),
        "k": problem.k,
        "objective_family": problem.objective_family,
        "solver_config": asdict(config) if config is not None else None,
        "overrun_seconds": result.overrun_seconds,
        "reference_objective": reference,
        "reference_kind": instance["reference_kind"],
        "objective_scale": scale,
        "absolute_gap": gap,
        "normalised_gap": normalized,
        "feasible": result.candidate is not None,
        "target_hit": bool(target_times),
        "time_to_target_seconds": min(target_times) if target_times else None,
        "reference_contradiction": normalized is not None and normalized < -1e-8,
    }


def summarise_budget_trials(trials):
    groups = defaultdict(lambda: defaultdict(list))
    for row in trials:
        groups[
            (
                row["n_assets"],
                row["k"],
                row.get("variant", row["method"]),
                row["method"],
                row["budget_seconds"],
            )
        ][row["problem_id"]].append(row)
    summaries, reliability = [], []
    for (n, k, variant, method, seconds), instances in groups.items():
        rows = [r for runs in instances.values() for r in runs]
        quality = [
            mean(r["normalised_gap"] for r in runs if r["normalised_gap"] is not None)
            for runs in instances.values()
            if any(r["normalised_gap"] is not None for r in runs)
        ]
        for problem_id, runs in instances.items():
            times = sorted(
                r["time_to_target_seconds"] if r["target_hit"] else math.inf
                for r in runs
            )
            # Empirical nearest-rank percentile includes every censored failure.
            t90 = times[math.ceil(0.9 * len(times)) - 1]
            reliability.append(
                {
                    "problem_id": problem_id,
                    "n_assets": n,
                    "k": k,
                    "method": method,
                    "variant": variant,
                    "budget_seconds": seconds,
                    "trial_count": len(runs),
                    "target_hit_rate": mean(r["target_hit"] for r in runs),
                    "t90_seconds": t90 if math.isfinite(t90) else None,
                }
            )
        summaries.append(
            {
                "n_assets": n,
                "k": k,
                "method": method,
                "variant": variant,
                "budget_seconds": seconds,
                "instance_count": len(instances),
                "trial_count": len(rows),
                "feasible_rate": mean(
                    mean(r["feasible"] for r in rs) for rs in instances.values()
                ),
                "target_hit_rate": mean(
                    mean(r["target_hit"] for r in rs) for rs in instances.values()
                ),
                "mean_normalised_gap_feasible": mean(quality) if quality else None,
                "quality_instance_count": len(quality),
                "time_limit_count": sum(r["status"] == "time_limit" for r in rows),
                "error_count": sum(r["status"] == "error" for r in rows),
                "reference_contradictions": sum(
                    r["reference_contradiction"] for r in rows
                ),
                "mean_elapsed_seconds": mean(r["elapsed_seconds"] for r in rows),
                "max_overrun_seconds": max(r["overrun_seconds"] for r in rows),
            }
        )
    return summaries, reliability


def paired_budget_trials(trials, trained="xy-dicke-ring-trained"):
    """Descriptive seed/instance pairs; missing and failed returns stay explicit."""
    grouped = defaultdict(dict)
    for row in trials:
        key = (row["problem_id"], row["seed"], row["budget_seconds"])
        variant = row.get("variant", row["method"])
        if variant in grouped[key]:
            raise ValueError("Duplicate paired trial")
        grouped[key][variant] = row
    pairs = []
    for (problem_id, seed, seconds), variants in grouped.items():
        if trained not in variants:
            continue
        first = variants[trained]
        for comparator, second in variants.items():
            if comparator == trained:
                continue
            both = first["feasible"] and second["feasible"]
            pairs.append(
                {
                    "problem_id": problem_id,
                    "n_assets": first["n_assets"],
                    "k": first["k"],
                    "seed": seed,
                    "budget_seconds": seconds,
                    "trained_variant": trained,
                    "comparator": comparator,
                    "trained_feasible": first["feasible"],
                    "comparator_feasible": second["feasible"],
                    "trained_target_hit": first["target_hit"],
                    "comparator_target_hit": second["target_hit"],
                    "both_feasible": both,
                    "normalised_improvement": (
                        second["normalised_gap"] - first["normalised_gap"]
                        if both
                        else None
                    ),
                }
            )
    return pairs
