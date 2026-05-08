from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

METHOD_COLORS = {
    "Classical exhaustive search": "#1f77b4",
    "Classical top-return heuristic": "#ff7f0e",
    "Classical minimum-variance subset": "#2ca02c",
    "Classical equal weight": "#7f7f7f",
    "Classical exact Markowitz": "#17becf",
    "Binary VQE best feasible": "#d62728",
    "QAOA X best feasible": "#9467bd",
    "QAOA XY best feasible": "#8c564b",
    "Fractional VQE": "#e377c2",
}


def _color_for_method(method: str, index: int = 0) -> str:
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    return cycle[index % len(cycle)]


def savefig(path: str | Path, dpi: int = 300) -> None:
    """
    Save the current matplotlib figure to `path`, ensuring the output directory exists.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=dpi, bbox_inches="tight")


def plot_trace(
    steps: Sequence[int],
    values: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: str | Path | None = None,
):
    fig = plt.figure()
    plt.plot(list(steps), list(values))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if outpath is not None:
        savefig(outpath)

    return fig


def bar_allocations(
    labels: Sequence[str],
    values: np.ndarray,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] = (0.0, 1.0),
    outpath: str | Path | None = None,
):
    fig = plt.figure()
    plt.bar(list(labels), np.array(values, dtype=float))
    plt.ylabel(ylabel)
    plt.ylim(*ylim)
    plt.title(title)
    plt.grid(axis="y")

    if outpath is not None:
        savefig(outpath)

    return fig


def plot_lambda_sweep_bars(
    lambdas: Sequence[float],
    mat: np.ndarray,  # shape (L, n)
    asset_labels: Sequence[str],
    ylabel: str,
    title: str,
    outpath: str | Path | None = None,
):
    mat = np.array(mat, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"mat must be 2D (L,n); got shape {mat.shape}")
    L, n = mat.shape
    if len(asset_labels) != n:
        raise ValueError(f"asset_labels length {len(asset_labels)} must match n={n}")
    if len(lambdas) != L:
        raise ValueError(f"lambdas length {len(lambdas)} must match L={L}")

    x = np.arange(L)
    bw = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(n):
        ax.bar(x + i * bw, mat[:, i], bw, label=asset_labels[i])

    # Numeric lambda ticks
    ax.set_xticks(x + bw * (n - 1) / 2)
    ax.set_xticklabels([f"{lam:.2f}" for lam in lambdas])

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Risk-aversion parameter λ")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y")

    # Annotations for "risky" and "safe"
    y_text = -0.12  # below x-axis

    ax.text(
        x[0] + bw * (n - 1) / 2,
        y_text,
        "risky",
        color="red",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=10,
        fontweight="bold",
    )

    ax.text(
        x[-1] + bw * (n - 1) / 2,
        y_text,
        "safe",
        color="green",
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=10,
        fontweight="bold",
    )

    if outpath is not None:
        savefig(outpath, dpi=200)

    return fig


def plot_frontier(
    risks: np.ndarray,
    returns: np.ndarray,
    lambdas_sorted: np.ndarray,
    title: str,
    outpath: str | Path | None = None,
):
    fig = plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        np.array(risks, dtype=float),
        np.array(returns, dtype=float),
        c=np.array(lambdas_sorted, dtype=float),
        cmap="plasma",
        s=50,
    )
    plt.plot(risks, returns, alpha=0.6)
    cbar = plt.colorbar(sc)
    cbar.set_label("λ")
    plt.xlabel("Portfolio risk σ")
    plt.ylabel("Expected return")
    plt.title(title)
    plt.grid(True)

    if outpath is not None:
        savefig(outpath, dpi=200)

    return fig


def plot_comparison_metric_bars(
    rows: Sequence[dict[str, str]],
    metric: str,
    title: str,
    ylabel: str | None = None,
    outpath: str | Path | None = None,
):
    labels = [row["method"] for row in rows]
    values = [float(row[metric]) for row in rows]
    colors = [_color_for_method(label, i) for i, label in enumerate(labels)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.35)
    ax.tick_params(axis="x", rotation=35)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(facecolor=color, label=label) for label, color in zip(labels, colors)
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
    )

    if outpath is not None:
        savefig(outpath, dpi=200)

    return fig


def plot_risk_return_comparison(
    rows: Sequence[dict[str, str]],
    title: str,
    outpath: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, row in enumerate(rows):
        risk = float(row.get("risk", row.get("mean_risk", 0.0)))
        ret = float(row.get("return", row.get("mean_return", 0.0)))
        ax.scatter(
            risk,
            ret,
            s=55,
            color=_color_for_method(row["method"], i),
            label=row["method"],
        )
    ax.set_xlabel("Portfolio risk σ")
    ax.set_ylabel("Expected return")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
    )

    if outpath is not None:
        savefig(outpath, dpi=200)

    return fig
