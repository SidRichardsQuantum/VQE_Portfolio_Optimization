"""Common reporting for independently evaluated classical and quantum candidates."""

from dataclasses import dataclass
from statistics import mean, pstdev

import numpy as np

from .problem import PortfolioProblem


@dataclass
class CandidateTrial:
    candidate: np.ndarray | None
    runtime: float
    seed: int | str = "-"
    sample_feasible_rate: float | None = None
    status: str = ""
    lower_bound: float | None = None
    mip_gap: float | None = None
    notes: str = ""


def _number(value):
    return "" if value is None else f"{value:.12g}"


def comparison_rows(
    problem: PortfolioProblem, trials, *, dataset, method, method_type, alpha=None
):
    """Ignore solver-reported objective/feasibility; evaluate the raw candidate.

    All trials contribute to runtime and feasibility. Only independently valid
    candidates contribute to quality statistics. Sample feasibility is separate.
    """
    if not trials:
        raise ValueError("At least one trial is required.")
    common = {
        "dataset": dataset,
        "n_assets": str(len(problem.mu)),
        "k": "" if problem.k is None else str(problem.k),
        "method": method,
        "type": method_type,
        "lambda": f"{problem.lam:.6g}",
        "objective_family": problem.objective_family,
        "reported_weighting": problem.reported_weighting,
        "problem_id": problem.problem_id,
    }
    evaluations, rows = [], []
    for trial in trials:
        evaluation = problem.evaluate(trial.candidate, alpha=alpha)
        evaluations.append(evaluation)
        if evaluation.feasible:
            values = np.asarray(trial.candidate, dtype=float)
            candidate = (
                "".join(str(int(x)) for x in values)
                if problem.k is not None
                else "[" + " ".join(f"{float(x):.17g}" for x in values) + "]"
            )
        else:
            candidate = "" if trial.candidate is None else repr(trial.candidate)
        rows.append(
            {
                **common,
                "seed": str(trial.seed),
                "objective": _number(evaluation.objective),
                "return": _number(evaluation.ret),
                "risk": _number(evaluation.risk),
                "feasible_rate": f"{float(evaluation.feasible):.6f}",
                "sample_feasible_rate": _number(trial.sample_feasible_rate),
                "runtime_seconds": _number(trial.runtime),
                "selection_or_weights": candidate,
                "penalized_energy": _number(evaluation.penalized_energy),
                "validation_error": evaluation.reason,
                "notes": trial.notes,
                "solver_status": trial.status,
                "lower_bound": _number(trial.lower_bound),
                "mip_gap": _number(trial.mip_gap),
            }
        )
    valid = [i for i, e in enumerate(evaluations) if e.feasible]
    objectives = [evaluations[i].objective for i in valid]
    sample_rates = [
        t.sample_feasible_rate for t in trials if t.sample_feasible_rate is not None
    ]
    summary = {
        **common,
        "seed_count": str(len(trials)),
        "best_objective": _number(min(objectives)) if valid else "",
        "mean_objective": _number(mean(objectives)) if valid else "",
        "std_objective": _number(pstdev(objectives)) if valid else "",
        "mean_return": (
            _number(mean(evaluations[i].ret for i in valid)) if valid else ""
        ),
        "mean_risk": _number(mean(evaluations[i].risk for i in valid)) if valid else "",
        "feasible_rate": f"{len(valid) / len(trials):.6f}",
        "sample_feasible_rate": _number(mean(sample_rates)) if sample_rates else "",
        "mean_runtime_seconds": _number(mean(t.runtime for t in trials)),
        "best_selection_or_weights": (
            rows[min(valid, key=lambda i: evaluations[i].objective)][
                "selection_or_weights"
            ]
            if valid
            else ""
        ),
        "notes": "Quality statistics include feasible candidates only; runtime and feasibility include all trials.",
        "solver_status": ";".join(sorted({t.status for t in trials if t.status})),
        "lower_bound": rows[0]["lower_bound"] if len(trials) == 1 else "",
        "mip_gap": rows[0]["mip_gap"] if len(trials) == 1 else "",
    }
    return summary, rows
