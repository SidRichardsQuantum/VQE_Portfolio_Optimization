from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .metrics import portfolio_return, portfolio_risk, symmetrize
from .utils import normalize_rows, topk_onehot


@dataclass(frozen=True)
class Frontier:
    risks: np.ndarray    # shape (L,)
    returns: np.ndarray  # shape (L,)
    lambdas: np.ndarray  # shape (L,) sorted by risk
    weights: np.ndarray  # shape (L, n) weights used to compute points


def _as_float_array(x) -> np.ndarray:
    return np.array(x, dtype=float)


def binary_frontier_from_probs(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lambdas: np.ndarray,
    probs_by_lambda: np.ndarray,
    k: int,
    *,
    equal_weight: bool = False,
    sort_by: Literal["risk", "lambda"] = "risk",
) -> Frontier:
    mu = _as_float_array(mu)
    Sigma = symmetrize(_as_float_array(Sigma))
    lambdas = _as_float_array(lambdas)
    probs = _as_float_array(probs_by_lambda)

    if probs.ndim != 2:
        raise ValueError(f"probs_by_lambda must be 2D (L,n); got shape {probs.shape}")
    L, n = probs.shape
    if len(mu) != n:
        raise ValueError(f"mu length {len(mu)} must match probs n={n}")
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma must be shape (n,n)=({n},{n}); got {Sigma.shape}")

    weights = np.zeros((L, n), dtype=float)
    rets = np.zeros(L, dtype=float)
    riks = np.zeros(L, dtype=float)

    for i in range(L):
        x = topk_onehot(probs[i], k=k)  # float 0/1
        if equal_weight:
            if k <= 0:
                raise ValueError("k must be positive when equal_weight=True")
            w = x / float(k)
        else:
            w = x
        weights[i] = w
        rets[i] = portfolio_return(mu, w)
        riks[i] = portfolio_risk(Sigma, w)

    if sort_by == "risk":
        order = np.argsort(riks)
    elif sort_by == "lambda":
        order = np.arange(L)
    else:
        raise ValueError(f"sort_by must be 'risk' or 'lambda'; got {sort_by}")

    return Frontier(
        risks=riks[order],
        returns=rets[order],
        lambdas=lambdas[order],
        weights=weights[order],
    )


def fractional_frontier_from_allocs(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lambdas: np.ndarray,
    allocs_by_lambda: np.ndarray,
    *,
    renormalize: bool = True,
    sort_by: Literal["risk", "lambda"] = "risk",
) -> Frontier:
    mu = _as_float_array(mu)
    Sigma = symmetrize(_as_float_array(Sigma))
    lambdas = _as_float_array(lambdas)
    allocs = _as_float_array(allocs_by_lambda)

    if allocs.ndim != 2:
        raise ValueError(f"allocs_by_lambda must be 2D (L,n); got shape {allocs.shape}")
    L, n = allocs.shape
    if len(mu) != n:
        raise ValueError(f"mu length {len(mu)} must match allocs n={n}")
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma must be shape (n,n)=({n},{n}); got {Sigma.shape}")

    weights = allocs.copy()
    if renormalize:
        weights = normalize_rows(weights)

    rets = np.array([portfolio_return(mu, weights[i]) for i in range(L)], dtype=float)
    riks = np.array([portfolio_risk(Sigma, weights[i]) for i in range(L)], dtype=float)

    if sort_by == "risk":
        order = np.argsort(riks)
    elif sort_by == "lambda":
        order = np.arange(L)
    else:
        raise ValueError(f"sort_by must be 'risk' or 'lambda'; got {sort_by}")

    return Frontier(
        risks=riks[order],
        returns=rets[order],
        lambdas=lambdas[order],
        weights=weights[order],
    )
