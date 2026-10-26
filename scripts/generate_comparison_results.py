#!/usr/bin/env python
"""Generate compact comparison CSVs for synthetic portfolio benchmarks.

The script intentionally uses deterministic synthetic data and NumPy-only
classical baselines so it can run without notebook execution, market-data
downloads, or optional solver dependencies. Opt into stronger solvers with
--methods classical-solvers and install .[classical].
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from vqe_portfolio import (
    BinaryVQEConfig,
    FractionalVQEConfig,
    QAOAConfig,
    run_binary_vqe,
    run_fractional_vqe,
    run_qaoa,
)
from vqe_portfolio import comparison as _comparison
from vqe_portfolio.benchmark import CandidateTrial, comparison_rows
from vqe_portfolio.classical import (
    convex_markowitz_baseline,
    local_search_binary_baseline,
    mixed_integer_binary_baseline,
)
from vqe_portfolio.comparison import (
    equal_weight_baseline,
    exact_long_only_markowitz_baseline,
    exhaustive_binary_baseline,
    feasible_sample_rate,
    minimum_variance_binary_baseline,
    top_return_baseline,
)
from vqe_portfolio.problem import PortfolioProblem

# Preserve helpers previously defined here for notebook/external callers.
binary_objective = _comparison.binary_objective
fractional_objective = _comparison.fractional_objective
selected_equal_weight = _comparison.selected_equal_weight
weights_string = _comparison.weights_string
bitstring = _comparison.bitstring

FIELDNAMES = [
    "dataset",
    "n_assets",
    "k",
    "method",
    "type",
    "lambda",
    "objective_family",
    "reported_weighting",
    "seed_count",
    "best_objective",
    "mean_objective",
    "std_objective",
    "mean_return",
    "mean_risk",
    "feasible_rate",
    "mean_runtime_seconds",
    "best_selection_or_weights",
    "notes",
    "solver_status",
    "lower_bound",
    "mip_gap",
    "problem_id",
    "sample_feasible_rate",
]

TRIAL_FIELDNAMES = [
    "dataset",
    "n_assets",
    "k",
    "method",
    "type",
    "lambda",
    "objective_family",
    "reported_weighting",
    "seed",
    "objective",
    "return",
    "risk",
    "feasible_rate",
    "runtime_seconds",
    "selection_or_weights",
    "notes",
    "solver_status",
    "lower_bound",
    "mip_gap",
    "problem_id",
    "sample_feasible_rate",
]

TRIAL_FIELDNAMES += ["penalized_energy", "validation_error"]


def synthetic_problem(n_assets: int, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + n_assets)
    mu = np.linspace(0.07, 0.18, n_assets) + rng.normal(0.0, 0.008, n_assets)
    factors = rng.normal(0.0, 0.08, size=(n_assets, min(3, n_assets)))
    Sigma = factors @ factors.T + np.diag(np.linspace(0.03, 0.09, n_assets))
    return mu.astype(float), Sigma.astype(float)


def _append(
    rows, trial_rows, problem, trials, dataset, method, method_type, alpha=None
):
    summary, runs = comparison_rows(
        problem,
        trials,
        dataset=dataset,
        method=method,
        method_type=method_type,
        alpha=alpha,
    )
    rows.append(summary)
    trial_rows.extend(runs)


def add_classical_rows(rows, trial_rows, dataset, mu, Sigma, lam, alpha, k):
    binary = PortfolioProblem(mu, Sigma, lam, "binary_qubo", k)
    fractional = PortfolioProblem(mu, Sigma, lam, "fractional_simplex")
    for method, baseline, problem, method_type in [
        (
            "Classical exhaustive search",
            exhaustive_binary_baseline,
            binary,
            "classical",
        ),
        (
            "Classical top-return heuristic",
            top_return_baseline,
            binary,
            "classical heuristic",
        ),
        (
            "Classical minimum-variance subset",
            minimum_variance_binary_baseline,
            binary,
            "classical heuristic",
        ),
        (
            "Classical equal weight",
            equal_weight_baseline,
            fractional,
            "classical heuristic",
        ),
        (
            "Classical exact Markowitz",
            exact_long_only_markowitz_baseline,
            fractional,
            "classical",
        ),
    ]:
        kwargs = {"alpha": alpha, "k": k} if problem.k is not None else {}
        started = time.perf_counter()
        candidate, _ = baseline(problem.mu, problem.Sigma, lam=problem.lam, **kwargs)
        trial = CandidateTrial(candidate, time.perf_counter() - started)
        _append(
            rows,
            trial_rows,
            problem,
            [trial],
            dataset,
            method,
            method_type,
            alpha if problem.k is not None else None,
        )


def add_solver_rows(
    rows, trial_rows, dataset, mu, Sigma, lam, k, seeds, time_limit=60.0, restarts=16
):
    binary = PortfolioProblem(mu, Sigma, lam, "binary_qubo", k)
    fractional = PortfolioProblem(mu, Sigma, lam, "fractional_simplex")
    for method, problem, run_seeds, solve in [
        (
            "Classical convex Markowitz (OSQP)",
            fractional,
            ["-"],
            lambda p, seed: convex_markowitz_baseline(p.mu, p.Sigma, p.lam),
        ),
        (
            "Classical mixed integer (HiGHS)",
            binary,
            ["-"],
            lambda p, seed: mixed_integer_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k, time_limit=time_limit
            ),
        ),
        (
            "Classical multistart swap search",
            binary,
            seeds,
            lambda p, seed: local_search_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k, seed=seed, restarts=restarts
            ),
        ),
    ]:
        trials = []
        for seed in run_seeds:
            started = time.perf_counter()
            result = solve(problem, seed)
            trials.append(
                CandidateTrial(
                    result.candidate,
                    time.perf_counter() - started,
                    seed=seed,
                    status=result.status,
                    lower_bound=result.lower_bound,
                    mip_gap=result.mip_gap,
                    notes=result.message,
                )
            )
        _append(
            rows,
            trial_rows,
            problem,
            trials,
            dataset,
            method,
            "classical heuristic" if method.endswith("search") else "classical",
        )


def _add_quantum_row(
    rows, trial_rows, dataset, mu, Sigma, cfg, seeds, solve, method, binary
):
    problem = PortfolioProblem(
        mu,
        Sigma,
        cfg.lam,
        "binary_qubo" if binary else "fractional_simplex",
        cfg.k if binary else None,
    )
    trials = []
    for seed in seeds:
        started = time.perf_counter()
        result = solve(problem.mu, problem.Sigma, replace(cfg, seed=seed))
        if binary:
            fallback = result.x_best_feasible is None
            candidate = result.x_topk if fallback else result.x_best_feasible
            sample_rate = feasible_sample_rate(result.sample_counts, problem.k)
            notes = "Top-K fallback" if fallback else "Best feasible sampled candidate"
        else:
            candidate, sample_rate, notes = (
                result.weights,
                None,
                "Returned simplex allocation",
            )
        trials.append(
            CandidateTrial(
                candidate,
                time.perf_counter() - started,
                seed=seed,
                sample_feasible_rate=sample_rate,
                notes=notes,
            )
        )
    _append(
        rows,
        trial_rows,
        problem,
        trials,
        dataset,
        method,
        "quantum",
        cfg.alpha if binary else None,
    )


def add_binary_vqe_row(rows, trial_rows, dataset, mu, Sigma, base_cfg, seeds):
    _add_quantum_row(
        rows,
        trial_rows,
        dataset,
        mu,
        Sigma,
        base_cfg,
        seeds,
        run_binary_vqe,
        "Binary VQE best feasible",
        True,
    )


def add_qaoa_row(rows, trial_rows, dataset, mu, Sigma, base_cfg, seeds):
    _add_quantum_row(
        rows,
        trial_rows,
        dataset,
        mu,
        Sigma,
        base_cfg,
        seeds,
        run_qaoa,
        f"QAOA {base_cfg.mixer.upper()} best feasible",
        True,
    )


def add_fractional_vqe_row(rows, trial_rows, dataset, mu, Sigma, base_cfg, seeds):
    _add_quantum_row(
        rows,
        trial_rows,
        dataset,
        mu,
        Sigma,
        base_cfg,
        seeds,
        run_fractional_vqe,
        "Fractional VQE",
        False,
    )


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_methods(value: str) -> set[str]:
    methods = {item.strip().lower() for item in value.split(",") if item.strip()}
    if "all" in methods:
        methods = (methods - {"all"}) | {
            "classical",
            "binary-vqe",
            "qaoa-x",
            "qaoa-xy",
            "fractional-vqe",
        }
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/generated_comparison_summary.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--trials-output",
        default="results/generated_repeatability_trials.csv",
        help="Per-seed trial CSV output path.",
    )
    parser.add_argument(
        "--asset-counts",
        default="4",
        help="Comma-separated synthetic problem sizes.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated quantum and local-search seeds.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated subset: classical,classical-solvers,binary-vqe,qaoa-x,qaoa-xy,fractional-vqe. all excludes optional classical-solvers.",
    )
    parser.add_argument(
        "--solver-time-limit",
        type=float,
        default=60.0,
        help="HiGHS solve limit in seconds (excludes model construction).",
    )
    parser.add_argument("--local-restarts", type=int, default=16)
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Selection size; default is min(2, n-1), at least 1.",
    )
    parser.add_argument("--lambda-value", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--binary-depth", type=int, default=2)
    parser.add_argument("--qaoa-p", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asset_counts = parse_int_list(args.asset_counts)
    seeds = parse_int_list(args.seeds)
    methods = parse_methods(args.methods)

    if not seeds or not asset_counts or any(n < 1 for n in asset_counts):
        raise ValueError("Provide seeds and positive asset counts.")
    allowed = {
        "classical",
        "classical-solvers",
        "binary-vqe",
        "qaoa-x",
        "qaoa-xy",
        "fractional-vqe",
    }
    if not methods or methods - allowed:
        raise ValueError("Unknown or empty method selection.")

    snapshots = {}
    rows: list[dict[str, str]] = []
    trial_rows: list[dict[str, str]] = []
    for n_assets in asset_counts:
        k = args.k if args.k is not None else max(1, min(2, n_assets - 1))
        if not 1 <= k <= n_assets:
            raise ValueError("k must be between 1 and each asset count.")
        dataset = f"Synthetic generated n={n_assets}"
        mu, Sigma = synthetic_problem(n_assets)
        for family, cardinality in [("binary_qubo", k), ("fractional_simplex", None)]:
            problem = PortfolioProblem(
                mu, Sigma, args.lambda_value, family, cardinality
            )
            snapshots[problem.problem_id] = problem.to_dict()

        if "classical" in methods:
            add_classical_rows(
                rows,
                trial_rows,
                dataset,
                mu,
                Sigma,
                args.lambda_value,
                args.alpha,
                k,
            )

        if "classical-solvers" in methods:
            add_solver_rows(
                rows,
                trial_rows,
                dataset,
                mu,
                Sigma,
                args.lambda_value,
                k,
                seeds,
                args.solver_time_limit,
                args.local_restarts,
            )

        if "binary-vqe" in methods:
            cfg = BinaryVQEConfig(
                depth=args.binary_depth,
                steps=args.steps,
                stepsize=0.3,
                log_every=max(args.steps, 1),
                lam=args.lambda_value,
                alpha=args.alpha,
                k=k,
                shots_train=None,
                shots_sample=args.shots,
            )
            add_binary_vqe_row(rows, trial_rows, dataset, mu, Sigma, cfg, seeds)

        if "qaoa-x" in methods:
            cfg = QAOAConfig(
                p=args.qaoa_p,
                steps=args.steps,
                stepsize=0.2,
                log_every=max(args.steps, 1),
                lam=args.lambda_value,
                alpha=args.alpha,
                k=k,
                mixer="x",
                shots_train=None,
                shots_sample=args.shots,
            )
            add_qaoa_row(rows, trial_rows, dataset, mu, Sigma, cfg, seeds)

        if "qaoa-xy" in methods:
            cfg = QAOAConfig(
                p=args.qaoa_p,
                steps=args.steps,
                stepsize=0.2,
                log_every=max(args.steps, 1),
                lam=args.lambda_value,
                alpha=args.alpha,
                k=k,
                mixer="xy",
                shots_train=None,
                shots_sample=args.shots,
            )
            add_qaoa_row(rows, trial_rows, dataset, mu, Sigma, cfg, seeds)

        if "fractional-vqe" in methods:
            cfg = FractionalVQEConfig(
                steps=args.steps,
                stepsize=0.3,
                log_every=max(args.steps, 1),
                lam=args.lambda_value,
                shots=None,
            )
            add_fractional_vqe_row(rows, trial_rows, dataset, mu, Sigma, cfg, seeds)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = output.with_suffix(".problems.json")
    snapshot_path.write_text(json.dumps(snapshots, indent=2, allow_nan=False) + "\n")
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    trials_output = Path(args.trials_output)
    trials_output.parent.mkdir(parents=True, exist_ok=True)
    with trials_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(trial_rows)

    print(f"Wrote {len(rows)} summary rows to {output}")
    print(f"Wrote {len(trial_rows)} trial rows to {trials_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
