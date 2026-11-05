"""Metrics and paired comparisons along uninterrupted training trajectories."""

from collections import defaultdict
from statistics import mean

import numpy as np

from .problem import PortfolioProblem
from .sampling_allowance import success_probability, validate_draw_counts
from .training_diagnostics import distribution_metrics, target_basis, train_checkpoints


def measure_trajectory(key, instance, method, seed, config, steps, draws, source=None):
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
        draws = validate_draw_counts(draws)
        problem = PortfolioProblem.from_dict(instance["snapshot"])
        if problem.problem_id != instance["problem_id"]:
            raise ValueError("Problem snapshot hash mismatch.")
        bits, feasible, target, objectives = target_basis(
            problem, instance["reference_objective"], config.target_gap
        )
        energies = objectives + config.alpha * (bits.sum(axis=1) - problem.k) ** 2
        result = train_checkpoints(problem, method, seed, config, steps)
        row.update(result)
        for state in row["checkpoints"]:
            metrics = distribution_metrics(
                state["probabilities"], feasible, target, energies
            )
            state.update(metrics)
            state["success_probabilities"] = {
                str(m): success_probability(metrics["p_target"], m) for m in draws
            }
        row["target_basis_indices"] = np.flatnonzero(target).tolist()
        row["uniform_feasible_p_target"] = float(target.sum() / feasible.sum())
        row["source_checkpoints_verified"] = []
        if source is not None and not source["error"]:
            source_step = source["algorithm_config"]["steps"]
            for state in row["checkpoints"]:
                phase = (
                    "before"
                    if state["step"] == 0
                    else "after" if state["step"] == source_step else None
                )
                if phase is None:
                    continue
                params = (
                    source["initial_params"]
                    if phase == "before"
                    else source["trained_params"]
                )
                if not np.allclose(
                    state["params"], params, rtol=1e-10, atol=1e-12
                ) or not np.allclose(
                    state["probabilities"],
                    source[f"{phase}_probabilities"],
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "Trajectory checkpoint disagrees with the saved initial/25-step diagnostic."
                    )
                row["source_checkpoints_verified"].append(state["step"])
    except Exception as exc:  # noqa: BLE001 - preserve failed trajectories
        row.pop("checkpoints", None)
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def checkpoint_rows(trajectories, steps, draws):
    rows = []
    for trial in trajectories:
        states = {s["step"]: s for s in trial.get("checkpoints", [])}
        for step in steps:
            state = states.get(step, {}) if not trial["error"] else {}
            row = {
                k: trial[k]
                for k in (
                    "trial_id",
                    "problem_id",
                    "instance_seed",
                    "n_assets",
                    "k",
                    "method",
                    "seed",
                    "error",
                )
            }
            row["step"] = step
            for field in (
                "p_target",
                "p_feasible",
                "p_target_given_feasible",
                "expected_penalized_energy",
                "training_runtime_seconds",
            ):
                row[field] = state.get(field)
            for m in draws:
                row[f"success_{m}"] = state.get("success_probabilities", {}).get(str(m))
            rows.append(row)
    return rows


def summarise_effort(trajectories, steps, draws):
    rows = checkpoint_rows(trajectories, steps, draws)
    fields = (
        "p_target",
        "p_feasible",
        "training_runtime_seconds",
        *(f"success_{m}" for m in draws),
    )
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(row["n_assets"], row["k"], row["method"], row["step"])][
            row["problem_id"]
        ].append(row)
    summaries = []
    for (n, k, method, step), instances in groups.items():
        all_rows = [r for group in instances.values() for r in group]
        valid = [[r for r in group if not r["error"]] for group in instances.values()]
        valid = [group for group in valid if group]
        summary = {
            "n_assets": n,
            "k": k,
            "method": method,
            "step": step,
            "instance_count": len(instances),
            "trajectory_count": len(all_rows),
            "successful_trajectories": sum(len(group) for group in valid),
            "quality_instance_count": len(valid),
            "error_count": sum(bool(r["error"]) for r in all_rows),
        }
        for field in fields:
            summary[f"mean_{field}"] = (
                mean(mean(r[field] for r in group) for group in valid)
                if valid
                else None
            )
        summaries.append(summary)
    paired = []
    for start, end in ((a, b) for i, a in enumerate(steps) for b in steps[i + 1 :]):
        changes = []
        for trial in trajectories:
            row = {
                k: trial[k]
                for k in ("trial_id", "problem_id", "n_assets", "k", "method", "error")
            }
            if not trial["error"]:
                states = {s["step"]: s for s in trial["checkpoints"]}
                for field in fields:
                    if field.startswith("success_"):
                        m = field.removeprefix("success_")
                        row[field] = (
                            states[end]["success_probabilities"][m]
                            - states[start]["success_probabilities"][m]
                        )
                    else:
                        row[field] = states[end][field] - states[start][field]
            changes.append(row)
        grouped = defaultdict(lambda: defaultdict(list))
        for row in changes:
            grouped[(row["n_assets"], row["k"], row["method"])][
                row["problem_id"]
            ].append(row)
        for (n, k, method), instances in grouped.items():
            valid = [
                [r for r in group if not r["error"]] for group in instances.values()
            ]
            valid = [group for group in valid if group]
            flat = [r for group in valid for r in group]
            summary = {
                "n_assets": n,
                "k": k,
                "method": method,
                "from_step": start,
                "to_step": end,
                "instance_count": len(instances),
                "quality_instance_count": len(valid),
                "trajectory_count": sum(len(group) for group in instances.values()),
                "successful_pairs": len(flat),
                "improved_pairs": sum(r["p_target"] > 1e-12 for r in flat),
                "worsened_pairs": sum(r["p_target"] < -1e-12 for r in flat),
            }
            for field in fields:
                summary[f"mean_delta_{field}"] = (
                    mean(mean(r[field] for r in group) for group in valid)
                    if valid
                    else None
                )
            paired.append(summary)
    return rows, summaries, paired
