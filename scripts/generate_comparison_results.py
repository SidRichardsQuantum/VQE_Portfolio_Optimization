#!/usr/bin/env python
"""Generate compact comparison CSVs for synthetic portfolio benchmarks.

The script intentionally uses deterministic synthetic data and NumPy-only
classical baselines so it can run without notebook execution, market-data
downloads, or optional solver dependencies. Opt into stronger solvers with
--methods classical-solvers and install .[classical].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vqe_portfolio import BinaryVQEConfig, FractionalVQEConfig, FVQEConfig, QAOAConfig
from vqe_portfolio.comparison import (
    binary_objective,
    bitstring,
    equal_weight_baseline,
    exact_long_only_markowitz_baseline,
    exhaustive_binary_baseline,
    feasible_sample_rate,
    fractional_objective,
    minimum_variance_binary_baseline,
    selected_equal_weight,
    top_return_baseline,
    weights_string,
)
from vqe_portfolio.problem import PortfolioProblem
from vqe_portfolio.research.comparison import (
    FIELDNAMES,
    TRIAL_FIELDNAMES,
    add_binary_vqe_row,
    add_classical_rows,
    add_fractional_vqe_row,
    add_fvqe_row,
    add_qaoa_row,
    add_solver_rows,
    synthetic_problem,
)
from vqe_portfolio.research.infrastructure import write_csv

__all__ = [
    "FIELDNAMES",
    "TRIAL_FIELDNAMES",
    "add_binary_vqe_row",
    "add_classical_rows",
    "add_fractional_vqe_row",
    "add_fvqe_row",
    "add_qaoa_row",
    "add_solver_rows",
    "binary_objective",
    "bitstring",
    "equal_weight_baseline",
    "exact_long_only_markowitz_baseline",
    "exhaustive_binary_baseline",
    "feasible_sample_rate",
    "fractional_objective",
    "main",
    "minimum_variance_binary_baseline",
    "selected_equal_weight",
    "synthetic_problem",
    "top_return_baseline",
    "weights_string",
]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_methods(value: str) -> set[str]:
    methods = {item.strip().lower() for item in value.split(",") if item.strip()}
    if "all" in methods:
        methods = (methods - {"all"}) | {
            "classical",
            "binary-vqe",
            "f-vqe",
            "qaoa-x",
            "qaoa-xy",
            "variational-allocator",
        }
    if "fractional-vqe" in methods:
        methods = (methods - {"fractional-vqe"}) | {"variational-allocator"}
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
        help="Comma-separated quantum and stochastic classical solver seeds.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated subset: classical,classical-solvers,osqp,pgd,frank-wolfe,highs,scip,qp-round,swap,anneal,tabu,memetic,ils,grasp,vns,binary-vqe,f-vqe,qaoa-x,qaoa-xy,variational-allocator (legacy: fractional-vqe). all excludes classical-solvers and individual solver IDs.",
    )
    parser.add_argument(
        "--solver-time-limit",
        type=float,
        default=60.0,
        help="HiGHS/SCIP solve limit in seconds (excludes model construction).",
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
        "osqp",
        "highs",
        "scip",
        "qp-round",
        "swap",
        "anneal",
        "tabu",
        "memetic",
        "ils",
        "grasp",
        "vns",
        "pgd",
        "frank-wolfe",
        "binary-vqe",
        "f-vqe",
        "qaoa-x",
        "qaoa-xy",
        "variational-allocator",
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

        solver_methods = methods & {
            "osqp",
            "highs",
            "scip",
            "qp-round",
            "swap",
            "anneal",
            "tabu",
            "memetic",
            "ils",
            "grasp",
            "vns",
            "pgd",
            "frank-wolfe",
        }
        if "classical-solvers" in methods or solver_methods:
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
                methods=None if "classical-solvers" in methods else solver_methods,
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

        if "f-vqe" in methods:
            cfg = FVQEConfig(
                depth=args.binary_depth,
                steps=args.steps,
                log_every=max(args.steps, 1),
                lam=args.lambda_value,
                alpha=args.alpha,
                k=k,
                shots_sample=args.shots,
            )
            add_fvqe_row(rows, trial_rows, dataset, mu, Sigma, cfg, seeds)

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

        if "variational-allocator" in methods:
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
    write_csv(output, rows, fieldnames=FIELDNAMES)

    trials_output = Path(args.trials_output)
    trials_output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(trials_output, trial_rows, fieldnames=TRIAL_FIELDNAMES)

    print(f"Wrote {len(rows)} summary rows to {output}")
    print(f"Wrote {len(trial_rows)} trial rows to {trials_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
