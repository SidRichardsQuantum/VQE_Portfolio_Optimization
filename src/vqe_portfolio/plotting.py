from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


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

    fig = plt.figure(figsize=(8, 5))
    for i in range(n):
        plt.bar(x + i * bw, mat[:, i], bw, label=asset_labels[i])

    plt.xticks(x + bw * (n - 1) / 2, [f"{lam:.2f}" for lam in lambdas])
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.xlabel("Risk-aversion parameter λ")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y")

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
    sc = plt.scatter(np.array(risks, dtype=float), np.array(returns, dtype=float), c=np.array(lambdas_sorted, dtype=float), cmap="plasma", s=50)
    plt.plot(risks, returns, alpha=0.6)
    cbar = plt.colorbar(sc)
    cbar.set_label("λ")
    plt.xlabel("Portfolio risk (σ)")
    plt.ylabel("Expected return")
    plt.title(title)
    plt.grid(True)

    if outpath is not None:
        savefig(outpath, dpi=200)

    return fig
