from __future__ import annotations

import numpy as np


def symmetrize(Sigma: np.ndarray) -> np.ndarray:
    return 0.5 * (Sigma + Sigma.T)


def portfolio_return(mu: np.ndarray, w: np.ndarray) -> float:
    return float(mu @ w)


def portfolio_variance(Sigma: np.ndarray, w: np.ndarray) -> float:
    return float(w @ Sigma @ w)


def portfolio_risk(Sigma: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(portfolio_variance(Sigma, w)))
