"""Classical portfolio solvers with explicit constraints and solver metadata.

Optional solvers are imported only when called. Install ``.[classical]`` for
OSQP/CVXPY and SciPy's HiGHS MILP interface.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassicalResult:
    candidate: np.ndarray | None
    objective: float | None
    status: str
    lower_bound: float | None = None
    mip_gap: float | None = None
    message: str = ""


def _problem(mu, Sigma, lam, k=None):
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    if mu.ndim != 1 or not mu.size or Sigma.shape != (mu.size, mu.size):
        raise ValueError("Expected nonempty mu and a matching square covariance.")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(Sigma)):
        raise ValueError("Problem data must be finite.")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative.")
    if not np.allclose(Sigma, Sigma.T, atol=1e-12, rtol=1e-10):
        raise ValueError("Covariance must be symmetric.")
    if k is not None and (
        not isinstance(k, (int, np.integer)) or not 0 <= k <= len(mu)
    ):
        raise ValueError("k must be an integer between zero and n_assets.")
    return mu, (Sigma + Sigma.T) / 2


def _objective(x, mu, Sigma, lam):
    return float(lam * x @ Sigma @ x - mu @ x)


def convex_markowitz_baseline(mu, Sigma, lam) -> ClassicalResult:
    """Solve the long-only convex QP; reject non-PSD input and bad output.

    Small numerical negative weights are clipped and renormalised before the
    objective is recomputed. This function's caller should time the whole call.
    Solver optimality is numerical, not an independently certified lower bound.
    """
    mu, Sigma = _problem(mu, Sigma, lam)
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "Install vqe-portfolio[classical] for the QP baseline."
        ) from exc

    w = cp.Variable(len(mu))
    problem = cp.Problem(
        cp.Minimize(lam * cp.quad_form(w, Sigma) - mu @ w),
        [w >= 0, cp.sum(w) == 1],
    )
    if not problem.is_dcp():
        raise ValueError("The fractional baseline requires a PSD covariance.")
    problem.solve(
        solver="OSQP",
        eps_abs=1e-9,
        eps_rel=1e-9,
        max_iter=100000,
        warm_start=False,
    )
    status = str(problem.status)
    if w.value is None:
        return ClassicalResult(None, None, status)
    candidate = np.asarray(w.value, dtype=float)
    if (
        not np.all(np.isfinite(candidate))
        or candidate.min() < -1e-8
        or abs(candidate.sum() - 1) > 1e-8
    ):
        return ClassicalResult(None, None, "invalid_candidate", message=status)
    candidate = np.maximum(candidate, 0)
    candidate /= candidate.sum()
    return ClassicalResult(
        candidate,
        _objective(candidate, mu, Sigma, lam),
        status,
        message="Clipped numerical negative weights and normalised; no certified bound",
    )


def mixed_integer_binary_baseline(
    mu, Sigma, lam, k, *, time_limit=60.0, mip_rel_gap=0.0
) -> ClassicalResult:
    """Solve exact-K selection with HiGHS and a sparse MILP linearisation.

    For i < j introduce y_ij = x_i*x_j using the three McCormick constraints.
    x is binary and y is continuous in [0, 1]. No penalty is required. The
    formulation has O(n**2) variables/constraints, including negative couplings.
    time_limit applies to the solver, not Python model construction.
    """
    mu, Sigma = _problem(mu, Sigma, lam, k)
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive.")
    if not np.isfinite(mip_rel_gap) or mip_rel_gap < 0:
        raise ValueError("mip_rel_gap must be finite and nonnegative.")
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise ImportError(
            "Install vqe-portfolio[classical] for the MILP baseline."
        ) from exc

    n = len(mu)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    size = n + len(pairs)
    c = np.concatenate(
        [lam * np.diag(Sigma) - mu, [2 * lam * Sigma[i, j] for i, j in pairs]]
    )
    A = lil_matrix((1 + 3 * len(pairs), size))
    A[0, :n] = 1
    lower = np.full(A.shape[0], -np.inf)
    upper = np.zeros(A.shape[0])
    lower[0] = upper[0] = k
    for offset, (i, j) in enumerate(pairs):
        y = n + offset
        row = 1 + 3 * offset
        A[row, y], A[row, i] = 1, -1
        A[row + 1, y], A[row + 1, j] = 1, -1
        A[row + 2, i], A[row + 2, j], A[row + 2, y] = 1, 1, -1
        upper[row + 2] = 1
    result = milp(
        c,
        integrality=np.r_[np.ones(n), np.zeros(len(pairs))],
        bounds=Bounds(np.zeros(size), np.ones(size)),
        constraints=LinearConstraint(A.tocsc(), lower, upper),
        options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap},
    )
    status = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded"}.get(
        result.status, "solver_error"
    )
    candidate = None
    if result.x is not None:
        raw = result.x[:n]
        rounded = np.rint(raw)
        if (
            np.all(np.isfinite(raw))
            and np.all(abs(raw - rounded) <= 1e-6)
            and np.all((rounded == 0) | (rounded == 1))
            and rounded.sum() == k
        ):
            candidate = rounded.astype(int)
        else:
            status = "invalid_candidate"
    return ClassicalResult(
        candidate,
        None if candidate is None else _objective(candidate, mu, Sigma, lam),
        status,
        lower_bound=getattr(result, "mip_dual_bound", None),
        mip_gap=getattr(result, "mip_gap", None),
        message=str(result.message),
    )


def local_search_binary_baseline(
    mu, Sigma, lam, k, *, seed=0, restarts=16, max_swaps=1000
) -> ClassicalResult:
    """Multistart best-improvement one-for-one swaps preserving cardinality.

    Starts include top returns, the smallest diagonal costs, and seeded random
    subsets. Delta evaluation uses the quadratic gradient rather than solving
    each neighbour from scratch. No global optimality guarantee is implied.
    """
    mu, Sigma = _problem(mu, Sigma, lam, k)
    if not isinstance(restarts, int) or restarts < 2:
        raise ValueError("restarts must be an integer >= 2.")
    if not isinstance(max_swaps, int) or max_swaps < 1:
        raise ValueError("max_swaps must be a positive integer.")
    rng = np.random.default_rng(seed)
    Q = lam * Sigma
    starts = [np.argsort(-mu)[:k], np.argsort(np.diag(Q) - mu)[:k]]
    starts.extend(rng.choice(len(mu), k, replace=False) for _ in range(restarts - 2))
    best, best_obj = None, np.inf
    capped = False
    for selected in starts:
        x = np.zeros(len(mu), dtype=int)
        x[selected] = 1
        for _ in range(max_swaps):
            inside, outside = np.flatnonzero(x), np.flatnonzero(1 - x)
            if not len(inside) or not len(outside):
                break
            gradient = 2 * Q @ x - mu
            delta = (
                gradient[outside][None, :]
                - gradient[inside][:, None]
                + np.diag(Q)[inside][:, None]
                + np.diag(Q)[outside][None, :]
                - 2 * Q[np.ix_(inside, outside)]
            )
            a, b = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[a, b] >= -1e-12:
                break
            x[inside[a]], x[outside[b]] = 0, 1
        else:
            capped = True
        obj = _objective(x, mu, Sigma, lam)
        if obj < best_obj:
            best, best_obj = x.copy(), obj
    return ClassicalResult(
        best,
        best_obj,
        "swap_limit" if capped else "local_optimum",
        message=f"seed={seed}; restarts={restarts}; max_swaps={max_swaps}",
    )
