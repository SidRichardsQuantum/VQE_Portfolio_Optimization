from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np

T = TypeVar("T")


def set_global_seed(seed: int) -> None:
    """Seed NumPy and Python random for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_notebook_outdir(images_dir: str = "images", notebooks_dir: str = "notebooks") -> Path:
    """
    Resolve a notebook output directory that works when run from repo root or notebooks/.
    Returns a Path that exists.
    """
    if Path(notebooks_dir).exists() and not Path(images_dir).exists():
        return ensure_dir(Path(notebooks_dir) / images_dir)
    return ensure_dir(Path(images_dir))


def normalize_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize 1D array to sum to 1 with epsilon protection."""
    arr = np.array(v, dtype=float)
    s = float(np.sum(arr))
    return arr / (s + eps)


def normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize a 2D array so each row sums to 1."""
    m = np.array(mat, dtype=float)
    den = m.sum(axis=1, keepdims=True)
    return m / (den + eps)


def ensure_list(xs: Iterable[T]) -> list[T]:
    """Convert an iterable to a list."""
    return list(xs)


def topk_onehot(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return a 0/1 float vector with exactly k ones at the largest entries of `scores`.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1:
        raise ValueError(f"scores must be 1D; got shape {s.shape}")
    if k <= 0:
        raise ValueError("k must be positive")
    if k > s.size:
        raise ValueError(f"k={k} cannot exceed number of assets n={s.size}")

    idx = np.argsort(-s)[:k]
    x = np.zeros_like(s, dtype=float)
    x[idx] = 1.0
    return x
