#!/usr/bin/env python
"""Generate a larger real-data comparison example.

This script intentionally keeps the binary/QAOA universe at 12 assets. Those
methods simulate one qubit per asset, so the example is large enough to show
scaling behavior without turning the notebook into a long-running benchmark.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import nbformat as nbf  # noqa: E402
import numpy as np  # noqa: E402

from scripts.generate_comparison_results import (  # noqa: E402
    FIELDNAMES,
    add_binary_vqe_row,
    add_classical_rows,
    add_fractional_vqe_row,
    add_qaoa_row,
)
from vqe_portfolio import BinaryVQEConfig, FractionalVQEConfig, QAOAConfig  # noqa: E402
from vqe_portfolio.data import get_stock_data  # noqa: E402
from vqe_portfolio.plotting import (  # noqa: E402
    plot_comparison_metric_bars,
    plot_risk_return_comparison,
    savefig,
)

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "XOM",
    "JNJ",
    "PG",
    "HD",
    "UNH",
]


def _bitstring_to_labels(tickers: list[str], bitstring: str) -> str:
    return ", ".join(ticker for ticker, bit in zip(tickers, bitstring) if bit == "1")


def _add_selected_labels(rows: list[dict[str, str]], tickers: list[str]) -> None:
    for row in rows:
        candidate = row["best_selection_or_weights"]
        if row["objective_family"] == "binary_qubo":
            row["selected_assets"] = _bitstring_to_labels(tickers, candidate)
        else:
            row["selected_assets"] = ""


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = FIELDNAMES + ["selected_assets"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_top_returns(tickers: list[str], mu: np.ndarray, outpath: Path) -> None:
    order = np.argsort(mu)
    labels = [tickers[i] for i in order]
    values = np.asarray(mu, dtype=float)[order]
    colors = ["#4c78a8" if v >= 0 else "#f58518" for v in values]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, values, color=colors)
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_xlabel("Annualized expected return")
    ax.set_title("Larger Real-Data Universe: Asset Return Estimates")
    ax.grid(axis="x", alpha=0.35)
    savefig(outpath, dpi=200)
    plt.close(fig)


def _plot_fractional_weights(
    tickers: list[str],
    rows: list[dict[str, str]],
    outpath: Path,
) -> None:
    weight_rows = [
        row
        for row in rows
        if row["objective_family"] == "fractional_simplex"
        and row["method"] in {"Classical exact Markowitz", "Fractional VQE"}
    ]
    labels = [row["method"] for row in weight_rows]
    weights = []
    for row in weight_rows:
        raw = row["best_selection_or_weights"].strip("[]")
        weights.append(np.array([float(x) for x in raw.split()], dtype=float))

    def autopct(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= 3.0 else ""

    fig, axes = plt.subplots(1, len(labels), figsize=(12, 5))
    if len(labels) == 1:
        axes = [axes]

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = [palette[i % len(palette)] for i in range(len(tickers))]

    for ax, method, weight in zip(axes, labels, weights):
        ax.pie(
            weight,
            labels=None,
            autopct=autopct,
            startangle=90,
            counterclock=False,
            colors=colors,
            pctdistance=0.72,
            textprops={"fontsize": 8},
        )
        ax.set_title(method)
        ax.axis("equal")

    fig.suptitle("Larger Real-Data Universe: Fractional Allocation Pies")
    fig.legend(
        tickers,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        frameon=True,
    )
    savefig(outpath, dpi=200)
    plt.close(fig)


def _make_plots(rows: list[dict[str, str]], tickers: list[str], mu: np.ndarray) -> None:
    outdir = ROOT / "notebooks" / "examples" / "images"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_risk_return_comparison(
        rows,
        "Larger Real-Data Universe: Risk/Return Comparison",
        outdir / "Larger_Real_Data_Risk_Return.png",
    )
    plt.close()

    binary_rows = [row for row in rows if row["objective_family"] == "binary_qubo"]
    plot_comparison_metric_bars(
        binary_rows,
        "best_objective",
        "Larger Real-Data Universe: Binary Objective",
        "Binary QUBO objective",
        outdir / "Larger_Real_Data_Binary_Objective.png",
    )
    plt.close()

    fractional_rows = [
        row for row in rows if row["objective_family"] == "fractional_simplex"
    ]
    plot_comparison_metric_bars(
        fractional_rows,
        "best_objective",
        "Larger Real-Data Universe: Fractional Objective",
        "Simplex objective",
        outdir / "Larger_Real_Data_Fractional_Objective.png",
    )
    plt.close()

    plot_comparison_metric_bars(
        rows,
        "feasible_rate",
        "Larger Real-Data Universe: Feasibility",
        "Feasible sample/allocation rate",
        outdir / "Larger_Real_Data_Feasibility.png",
    )
    plt.close()

    _plot_top_returns(
        tickers,
        mu,
        outdir / "Larger_Real_Data_Asset_Returns.png",
    )
    _plot_fractional_weights(
        tickers,
        rows,
        outdir / "Larger_Real_Data_Fractional_Weights.png",
    )


def _make_notebook(
    *,
    tickers: list[str],
    start: str,
    end: str,
    lam: float,
    alpha: float,
    k: int,
    steps: int,
    shots: int,
    seed: int,
) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(dedent(f"""
                # Larger Real-Data Portfolio Example

                This notebook uses a 12-stock real-data universe to stress the
                package beyond the smaller examples. Binary VQE and QAOA use
                one qubit per asset, so this example is intentionally kept at
                12 assets rather than treated as a large-scale production run.

                Window: `{start}` to `{end}`  
                Tickers: `{", ".join(tickers)}`  
                Binary cardinality: `K={k}`  
                Risk-aversion: `lambda={lam}`  
                Penalty: `alpha={alpha}`
                """).strip()),
        nbf.v4.new_code_cell(dedent(f"""
                from pathlib import Path
                import sys

                import pandas as pd
                from IPython.display import Image, display

                ROOT = Path.cwd()
                if not (ROOT / "scripts").exists():
                    ROOT = ROOT.parents[1]
                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))

                from scripts.generate_larger_real_data_example import main

                main([
                    "--tickers", "{",".join(tickers)}",
                    "--start", "{start}",
                    "--end", "{end}",
                    "--lambda", "{lam}",
                    "--alpha", "{alpha}",
                    "--k", "{k}",
                    "--steps", "{steps}",
                    "--shots", "{shots}",
                    "--seed", "{seed}",
                ])
                """).strip()),
        nbf.v4.new_code_cell(dedent("""
                csv_path = ROOT / "results" / "larger_real_data_comparison.csv"
                rows = pd.read_csv(csv_path)
                rows[
                    [
                        "method",
                        "objective_family",
                        "k",
                        "mean_return",
                        "mean_risk",
                        "best_objective",
                        "feasible_rate",
                        "best_selection_or_weights",
                        "selected_assets",
                    ]
                ]
                """).strip()),
        nbf.v4.new_code_cell(dedent("""
                image_paths = [
                    "notebooks/examples/images/Larger_Real_Data_Risk_Return.png",
                    "notebooks/examples/images/Larger_Real_Data_Binary_Objective.png",
                    "notebooks/examples/images/Larger_Real_Data_Fractional_Objective.png",
                    "notebooks/examples/images/Larger_Real_Data_Feasibility.png",
                    "notebooks/examples/images/Larger_Real_Data_Asset_Returns.png",
                    "notebooks/examples/images/Larger_Real_Data_Fractional_Weights.png",
                ]

                for path in image_paths:
                    display(Image(filename=str(ROOT / path)))
                """).strip()),
        nbf.v4.new_markdown_cell(dedent("""
                The binary objective and fractional objective are different
                mathematical objectives. Compare methods within each objective
                family, and use the risk/return chart as a cross-family view of
                the reported portfolios.
                """).strip()),
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    outpath = ROOT / "notebooks" / "examples" / "04_Larger_Real_Data_Example.ipynb"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, outpath)


def generate(
    *,
    tickers: list[str],
    start: str,
    end: str,
    lam: float,
    alpha: float,
    k: int,
    steps: int,
    shots: int,
    seed: int,
) -> list[dict[str, str]]:
    mu_series, sigma_df, _prices = get_stock_data(
        tickers,
        start=start,
        end=end,
        use_log=True,
        shrink="lw",
        scale="none",
        progress=False,
    )
    mu = mu_series.values.astype(float)
    Sigma = sigma_df.values.astype(float)

    rows: list[dict[str, str]] = []
    trial_rows: list[dict[str, str]] = []
    dataset = f"Larger real data n={len(tickers)}"

    add_classical_rows(rows, trial_rows, dataset, mu, Sigma, lam, alpha, k)

    binary_cfg = BinaryVQEConfig(
        depth=1,
        steps=steps,
        stepsize=0.25,
        log_every=max(steps, 1),
        lam=lam,
        alpha=alpha,
        k=k,
        seed=seed,
        shots_train=None,
        shots_sample=shots,
    )
    add_binary_vqe_row(rows, trial_rows, dataset, mu, Sigma, binary_cfg, [seed])

    for mixer in ["x", "xy"]:
        qaoa_cfg = QAOAConfig(
            p=1,
            steps=steps,
            stepsize=0.15,
            log_every=max(steps, 1),
            lam=lam,
            alpha=alpha,
            k=k,
            mixer=mixer,
            seed=seed,
            shots_train=None,
            shots_sample=shots,
        )
        add_qaoa_row(rows, trial_rows, dataset, mu, Sigma, qaoa_cfg, [seed])

    fractional_cfg = FractionalVQEConfig(
        depth=2,
        steps=steps,
        stepsize=0.25,
        log_every=max(steps, 1),
        lam=lam,
        seed=seed,
        shots=None,
    )
    add_fractional_vqe_row(rows, trial_rows, dataset, mu, Sigma, fractional_cfg, [seed])

    _add_selected_labels(rows, tickers)
    _write_rows(ROOT / "results" / "larger_real_data_comparison.csv", rows)
    _make_plots(rows, tickers, mu)
    _make_notebook(
        tickers=tickers,
        start=start,
        end=end,
        lam=lam,
        alpha=alpha,
        k=k,
        steps=steps,
        shots=shots,
        seed=seed,
    )

    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--lambda", dest="lam", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict[str, str]]:
    args = parse_args(argv)
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",")]
    return generate(
        tickers=tickers,
        start=args.start,
        end=args.end,
        lam=args.lam,
        alpha=args.alpha,
        k=args.k,
        steps=args.steps,
        shots=args.shots,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
