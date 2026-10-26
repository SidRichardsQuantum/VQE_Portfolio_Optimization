#!/usr/bin/env python
"""Generate compact ansatz comparison CSVs for Binary and Fractional VQE."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from scripts.generate_comparison_results import (  # noqa: E402
    add_binary_vqe_row,
    add_fractional_vqe_row,
    exact_long_only_markowitz_baseline,
    exhaustive_binary_baseline,
    synthetic_problem,
)
from vqe_portfolio import BinaryVQEConfig, FractionalVQEConfig  # noqa: E402
from vqe_portfolio.ansatz import BINARY_ANSATZES, FRACTIONAL_ANSATZES  # noqa: E402
from vqe_portfolio.benchmark import CandidateTrial, comparison_rows  # noqa: E402
from vqe_portfolio.plotting import (  # noqa: E402
    plot_comparison_metric_bars,
    plot_risk_return_comparison,
)
from vqe_portfolio.problem import PortfolioProblem  # noqa: E402

FIELDNAMES = [
    "dataset",
    "method",
    "ansatz",
    "depth",
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
    "problem_id",
    "sample_feasible_rate",
]


def _ansatz_summary(row, ansatz, depth):
    return {
        field: (
            str(ansatz)
            if field == "ansatz"
            else str(depth) if field == "depth" else row.get(field, "")
        )
        for field in FIELDNAMES
    }


def binary_ansatz_rows(mu, Sigma, seeds, *, depth, steps, shots, lam, alpha, k):
    problem = PortfolioProblem(mu, Sigma, lam, "binary_qubo", k)
    mu, Sigma = problem.mu, problem.Sigma
    dataset = f"Synthetic ansatz n={len(mu)}"
    started = time.perf_counter()
    candidate, _ = exhaustive_binary_baseline(mu, Sigma, lam, alpha, k)
    exact, _ = comparison_rows(
        problem,
        [CandidateTrial(candidate, time.perf_counter() - started)],
        dataset=dataset,
        method="Classical exhaustive search",
        method_type="classical",
        alpha=alpha,
    )
    rows = [_ansatz_summary(exact, "-", 0)]
    for ansatz in BINARY_ANSATZES:
        cfg = BinaryVQEConfig(
            ansatz=ansatz,
            depth=depth,
            steps=steps,
            stepsize=0.3,
            log_every=max(steps, 1),
            lam=lam,
            alpha=alpha,
            k=k,
            shots_train=None,
            shots_sample=shots,
        )
        summaries, trials = [], []
        add_binary_vqe_row(summaries, trials, dataset, mu, Sigma, cfg, seeds)
        summaries[0]["method"] = "Binary VQE"
        rows.append(_ansatz_summary(summaries[0], ansatz, depth))
    return rows


def fractional_ansatz_rows(mu, Sigma, seeds, *, depth, steps, lam):
    problem = PortfolioProblem(mu, Sigma, lam, "fractional_simplex")
    mu, Sigma = problem.mu, problem.Sigma
    dataset = f"Synthetic ansatz n={len(mu)}"
    started = time.perf_counter()
    candidate, _ = exact_long_only_markowitz_baseline(mu, Sigma, lam)
    exact, _ = comparison_rows(
        problem,
        [CandidateTrial(candidate, time.perf_counter() - started)],
        dataset=dataset,
        method="Classical exact Markowitz",
        method_type="classical",
    )
    rows = [_ansatz_summary(exact, "-", 0)]
    for ansatz in FRACTIONAL_ANSATZES:
        cfg = FractionalVQEConfig(
            ansatz=ansatz,
            depth=depth,
            steps=steps,
            stepsize=0.3,
            log_every=max(steps, 1),
            lam=lam,
            shots=None,
        )
        summaries, trials = [], []
        add_fractional_vqe_row(summaries, trials, dataset, mu, Sigma, cfg, seeds)
        rows.append(_ansatz_summary(summaries[0], ansatz, depth))
    return rows


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def save_plots(rows: list[dict[str, str]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    binary_rows = [row for row in rows if row["objective_family"] == "binary_qubo"]
    fractional_rows = [
        row for row in rows if row["objective_family"] == "fractional_simplex"
    ]

    plot_risk_return_comparison(
        rows,
        title="Ansatz comparison: risk vs return",
        outpath=plot_dir / "Ansatz_Comparison_Risk_Return.png",
    )
    plt.close()
    plot_comparison_metric_bars(
        binary_rows,
        metric="mean_objective",
        title="Binary VQE ansatz comparison",
        ylabel="Mean binary QUBO objective",
        outpath=plot_dir / "Ansatz_Comparison_Binary_Objective.png",
    )
    plt.close()
    plot_comparison_metric_bars(
        fractional_rows,
        metric="mean_objective",
        title="Fractional VQE ansatz comparison",
        ylabel="Mean simplex objective",
        outpath=plot_dir / "Ansatz_Comparison_Fractional_Objective.png",
    )
    plt.close()
    plot_comparison_metric_bars(
        rows,
        metric="feasible_rate",
        title="Ansatz comparison: feasibility",
        ylabel="Feasible returned-portfolio rate",
        outpath=plot_dir / "Ansatz_Comparison_Feasibility.png",
    )
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/ansatz_comparison.csv")
    parser.add_argument("--plot-dir", default="notebooks/images")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--asset-count", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--lambda-value", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--k", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = parse_int_list(args.seeds)
    mu, Sigma = synthetic_problem(args.asset_count)
    k = max(1, min(args.k, args.asset_count - 1))

    rows = [
        *binary_ansatz_rows(
            mu,
            Sigma,
            seeds,
            depth=args.depth,
            steps=args.steps,
            shots=args.shots,
            lam=args.lambda_value,
            alpha=args.alpha,
            k=k,
        ),
        *fractional_ansatz_rows(
            mu,
            Sigma,
            seeds,
            depth=args.depth,
            steps=args.steps,
            lam=args.lambda_value,
        ),
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    if not args.skip_plots:
        save_plots(rows, Path(args.plot_dir))

    print(f"Wrote {len(rows)} ansatz comparison rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
