#!/usr/bin/env python
"""Generate compact ansatz comparison CSVs for Binary and Fractional VQE."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from scripts.generate_comparison_results import (  # noqa: E402
    binary_objective,
    exact_long_only_markowitz_baseline,
    exhaustive_binary_baseline,
    feasible_sample_rate,
    fractional_objective,
    selected_equal_weight,
    synthetic_problem,
    weights_string,
)
from vqe_portfolio import BinaryVQEConfig, FractionalVQEConfig  # noqa: E402
from vqe_portfolio.ansatz import BINARY_ANSATZES, FRACTIONAL_ANSATZES  # noqa: E402
from vqe_portfolio.binary import run_binary_vqe  # noqa: E402
from vqe_portfolio.fractional import run_fractional_vqe  # noqa: E402
from vqe_portfolio.metrics import portfolio_return, portfolio_risk  # noqa: E402
from vqe_portfolio.plotting import (  # noqa: E402
    plot_comparison_metric_bars,
    plot_risk_return_comparison,
)

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
]


def bitstring(x: np.ndarray) -> str:
    return "".join(str(int(v)) for v in np.asarray(x, dtype=int))


def summarize(
    *,
    dataset: str,
    method: str,
    ansatz: str,
    depth: int,
    objective_family: str,
    reported_weighting: str,
    objectives: list[float],
    returns: list[float],
    risks: list[float],
    feasible_rates: list[float],
    runtimes: list[float],
    candidates: list[str],
    notes: str,
) -> dict[str, str]:
    best_idx = int(np.argmin(objectives))
    return {
        "dataset": dataset,
        "method": method,
        "ansatz": ansatz,
        "depth": str(depth),
        "objective_family": objective_family,
        "reported_weighting": reported_weighting,
        "seed_count": str(len(objectives)),
        "best_objective": f"{min(objectives):.6f}",
        "mean_objective": f"{mean(objectives):.6f}",
        "std_objective": (
            f"{pstdev(objectives):.6f}" if len(objectives) > 1 else "0.000000"
        ),
        "mean_return": f"{mean(returns):.6f}",
        "mean_risk": f"{mean(risks):.6f}",
        "feasible_rate": f"{mean(feasible_rates):.6f}",
        "mean_runtime_seconds": f"{mean(runtimes):.6f}",
        "best_selection_or_weights": candidates[best_idx],
        "notes": notes,
    }


def binary_ansatz_rows(
    mu: np.ndarray,
    Sigma: np.ndarray,
    seeds: list[int],
    *,
    depth: int,
    steps: int,
    shots: int,
    lam: float,
    alpha: float,
    k: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    x_exact, exact_obj = exhaustive_binary_baseline(
        mu, Sigma, lam=lam, alpha=alpha, k=k
    )
    w_exact = selected_equal_weight(x_exact)
    rows.append(
        summarize(
            dataset=f"Synthetic ansatz n={len(mu)}",
            method="Classical exhaustive search",
            ansatz="-",
            depth=0,
            objective_family="binary_qubo",
            reported_weighting="equal_weight_selected",
            objectives=[exact_obj],
            returns=[portfolio_return(mu, w_exact)],
            risks=[portfolio_risk(Sigma, w_exact)],
            feasible_rates=[1.0],
            runtimes=[0.0],
            candidates=[bitstring(x_exact)],
            notes="Exact binary cardinality baseline",
        )
    )

    for ansatz in BINARY_ANSATZES:
        objectives: list[float] = []
        returns: list[float] = []
        risks: list[float] = []
        feasible_rates: list[float] = []
        runtimes: list[float] = []
        candidates: list[str] = []
        base_cfg = BinaryVQEConfig(
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
        for seed in seeds:
            started = time.perf_counter()
            res = run_binary_vqe(mu, Sigma, replace(base_cfg, seed=seed))
            runtimes.append(time.perf_counter() - started)
            x = np.asarray(
                res.x_best_feasible if res.x_best_feasible is not None else res.x_topk,
                dtype=int,
            )
            w = selected_equal_weight(x)
            objectives.append(binary_objective(x, mu, Sigma, lam=lam, alpha=alpha, k=k))
            returns.append(portfolio_return(mu, w))
            risks.append(portfolio_risk(Sigma, w))
            feasible_rates.append(feasible_sample_rate(res.sample_counts, k))
            candidates.append(bitstring(x))

        rows.append(
            summarize(
                dataset=f"Synthetic ansatz n={len(mu)}",
                method="Binary VQE",
                ansatz=ansatz,
                depth=depth,
                objective_family="binary_qubo",
                reported_weighting="equal_weight_selected",
                objectives=objectives,
                returns=returns,
                risks=risks,
                feasible_rates=feasible_rates,
                runtimes=runtimes,
                candidates=candidates,
                notes="Best feasible sampled candidate, falling back to Top-K projection",
            )
        )
    return rows


def fractional_ansatz_rows(
    mu: np.ndarray,
    Sigma: np.ndarray,
    seeds: list[int],
    *,
    depth: int,
    steps: int,
    lam: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    w_exact, exact_obj = exact_long_only_markowitz_baseline(mu, Sigma, lam=lam)
    rows.append(
        summarize(
            dataset=f"Synthetic ansatz n={len(mu)}",
            method="Classical exact Markowitz",
            ansatz="-",
            depth=0,
            objective_family="fractional_simplex",
            reported_weighting="simplex_weights",
            objectives=[exact_obj],
            returns=[portfolio_return(mu, w_exact)],
            risks=[portfolio_risk(Sigma, w_exact)],
            feasible_rates=[1.0],
            runtimes=[0.0],
            candidates=[weights_string(w_exact)],
            notes="Exact long-only simplex baseline from active-set enumeration",
        )
    )

    for ansatz in FRACTIONAL_ANSATZES:
        objectives: list[float] = []
        returns: list[float] = []
        risks: list[float] = []
        feasible_rates: list[float] = []
        runtimes: list[float] = []
        candidates: list[str] = []
        base_cfg = FractionalVQEConfig(
            ansatz=ansatz,
            depth=depth,
            steps=steps,
            stepsize=0.3,
            log_every=max(steps, 1),
            lam=lam,
            shots=None,
        )
        for seed in seeds:
            started = time.perf_counter()
            res = run_fractional_vqe(mu, Sigma, replace(base_cfg, seed=seed))
            runtimes.append(time.perf_counter() - started)
            w = np.asarray(res.weights, dtype=float)
            objectives.append(fractional_objective(w, mu, Sigma, lam=lam))
            returns.append(portfolio_return(mu, w))
            risks.append(portfolio_risk(Sigma, w))
            feasible_rates.append(
                float(np.all(w >= -1e-8) and abs(w.sum() - 1.0) < 1e-6)
            )
            candidates.append(weights_string(w))

        rows.append(
            summarize(
                dataset=f"Synthetic ansatz n={len(mu)}",
                method="Fractional VQE",
                ansatz=ansatz,
                depth=depth,
                objective_family="fractional_simplex",
                reported_weighting="simplex_weights",
                objectives=objectives,
                returns=returns,
                risks=risks,
                feasible_rates=feasible_rates,
                runtimes=runtimes,
                candidates=candidates,
                notes="Simplex-normalized allocation",
            )
        )
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
        ylabel="Feasible sample/simplex rate",
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
