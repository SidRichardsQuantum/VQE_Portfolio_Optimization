from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: str | Path, dpi: int = 300) -> None:
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_trace(steps: Sequence[int], values: Sequence[float], xlabel: str, ylabel: str, title: str, outpath: str | Path | None = None):
    plt.figure()
    plt.plot(list(steps), list(values))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if outpath is not None:
        savefig(outpath)
    return plt.gcf()


def bar_allocations(labels: Sequence[str], values: np.ndarray, ylabel: str, title: str, ylim=(0, 1), outpath: str | Path | None = None):
    plt.figure()
    plt.bar(list(labels), np.array(values, dtype=float))
    plt.ylabel(ylabel)
    plt.ylim(*ylim)
    plt.title(title)
    plt.grid(axis="y")
    if outpath is not None:
        savefig(outpath)
    return plt.gcf()


def plot_lambda_sweep_bars(
    lambdas: Sequence[float],
    mat: np.ndarray,  # shape (L, n)
    asset_labels: Sequence[str],
    ylabel: str,
    title: str,
    outpath: str | Path | None = None,
):
    L, n = mat.shape
    x = np.arange(L)
    bw = 0.8 / n

    plt.figure(figsize=(8, 5))
    for i in range(n):
        plt.bar(x + i * bw, mat[:, i], bw, label=asset_labels[i])

    plt.xticks(x + bw * (n - 1) / 2, [f"{l:.2f}" for l in lambdas])
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.xlabel("Risk-aversion parameter λ")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y")

    if outpath is not None:
        savefig(outpath, dpi=200)
    return plt.gcf()


def plot_frontier(
    risks: np.ndarray,
    returns: np.ndarray,
    lambdas_sorted: np.ndarray,
    title: str,
    outpath: str | Path | None = None,
):
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(risks, returns, c=lambdas_sorted, cmap="plasma", s=50)
    plt.plot(risks, returns, alpha=0.6)
    cbar = plt.colorbar(sc)
    cbar.set_label("λ")
    plt.xlabel("Portfolio risk (σ)")
    plt.ylabel("Expected return")
    plt.title(title)
    plt.grid(True)
    if outpath is not None:
        savefig(outpath, dpi=200)
    return plt.gcf()
