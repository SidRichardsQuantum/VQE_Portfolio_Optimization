"""NumPy reference solvers and simple portfolio heuristics."""

import itertools

import numpy as np

from ..problem import PortfolioProblem, binary_objective, fractional_objective


def exhaustive_binary_baseline(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    alpha: float,
    k: int,
) -> tuple[np.ndarray, float]:
    best_x: np.ndarray | None = None
    best_obj = float("inf")
    for combo in itertools.combinations(range(len(mu)), k):
        x = np.zeros(len(mu), dtype=int)
        x[list(combo)] = 1
        obj = binary_objective(x, mu, Sigma, lam=lam, alpha=alpha, k=k)
        if obj < best_obj:
            best_x = x
            best_obj = obj
    if best_x is None:
        raise RuntimeError("No feasible binary portfolio was generated.")
    return best_x, best_obj


def top_return_baseline(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    alpha: float,
    k: int,
) -> tuple[np.ndarray, float]:
    """Select the K largest returns, including the empty portfolio for K=0."""
    problem = PortfolioProblem(mu, Sigma, lam, "binary_qubo", k)
    mu, Sigma, k = problem.mu, problem.Sigma, problem.k
    selected = np.argsort(mu)[-k:] if k else np.array([], dtype=int)
    x = np.zeros(len(mu), dtype=int)
    x[selected] = 1
    return x, binary_objective(x, mu, Sigma, lam=lam, alpha=alpha, k=k)


def minimum_variance_binary_baseline(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    alpha: float,
    k: int,
) -> tuple[np.ndarray, float]:
    best_x: np.ndarray | None = None
    best_variance = float("inf")
    for combo in itertools.combinations(range(len(mu)), k):
        x = np.zeros(len(mu), dtype=int)
        x[list(combo)] = 1
        variance = float(x @ Sigma @ x)
        if variance < best_variance:
            best_x = x
            best_variance = variance
    if best_x is None:
        raise RuntimeError("No feasible binary portfolio was generated.")
    return best_x, binary_objective(best_x, mu, Sigma, lam=lam, alpha=alpha, k=k)


def exact_long_only_markowitz_baseline(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    tol: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """Solve the long-only simplex Markowitz problem by active-set enumeration.

    This is intended for small benchmark/client examples. It exactly solves the
    convex quadratic objective over every possible positive support, then keeps
    the best feasible KKT candidate.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)

    if lam <= 0:
        w = np.zeros(n, dtype=float)
        w[int(np.argmax(mu))] = 1.0
        return w, fractional_objective(w, mu, Sigma, lam=lam)

    best_w: np.ndarray | None = None
    best_obj = float("inf")

    for size in range(1, n + 1):
        for support in itertools.combinations(range(n), size):
            idx = np.array(support, dtype=int)
            sigma_s = Sigma[np.ix_(idx, idx)]
            mu_s = mu[idx]
            lhs = np.block(
                [
                    [2.0 * lam * sigma_s, np.ones((size, 1))],
                    [np.ones((1, size)), np.zeros((1, 1))],
                ]
            )
            rhs = np.concatenate([mu_s, np.array([1.0])])
            try:
                sol = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            w_s = sol[:size]
            if np.any(w_s < -tol):
                continue
            w_s = np.maximum(w_s, 0.0)
            total = float(w_s.sum())
            if total <= 0:
                continue
            w_s = w_s / total

            w = np.zeros(n, dtype=float)
            w[idx] = w_s
            obj = fractional_objective(w, mu, Sigma, lam=lam)
            if obj < best_obj:
                best_w = w
                best_obj = obj

    if best_w is None:
        raise RuntimeError("No feasible Markowitz allocation was generated.")

    return best_w, best_obj


def equal_weight_baseline(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, float]:
    w = np.full(len(mu), 1.0 / len(mu), dtype=float)
    return w, fractional_objective(w, mu, Sigma, lam=lam)
