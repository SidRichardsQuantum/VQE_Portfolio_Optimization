"""Classical portfolio solvers with explicit constraints and solver metadata.

Swap search, simulated annealing, and tabu search need only NumPy. Optional
solvers are imported only when called. Install ``.[classical]`` for OSQP/CVXPY
and SciPy's HiGHS MILP interface, plus PySCIPOpt's quadratic integer solver.
"""

from dataclasses import dataclass

import numpy as np

from ..problem import PortfolioProblem
from ..problem import portfolio_objective as _objective


@dataclass
class ClassicalResult:
    candidate: np.ndarray | None
    objective: float | None
    status: str
    lower_bound: float | None = None
    mip_gap: float | None = None
    message: str = ""


def _problem(mu, Sigma, lam, k=None):
    problem = PortfolioProblem(
        mu, Sigma, lam, "binary_qubo" if k is not None else "fractional_simplex", k
    )
    return problem.mu, problem.Sigma


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


def scip_binary_baseline(
    mu, Sigma, lam, k, *, time_limit=60.0, mip_rel_gap=0.0, candidate_callback=None
) -> ClassicalResult:
    """Solve exact-K selection with SCIP's quadratic constraint handler.

    Minimise an unrestricted epigraph variable z subject to f(x) <= z, with
    binary x and sum(x) == k. Unlike the HiGHS MILP, no pairwise product
    variables are introduced. Signed couplings and negative objectives are
    supported. Optional PySCIPOpt is imported only when called.

    The native time limit excludes model construction and Python cleanup.
    Native termination status, finite numerical dual bound, and SCIP's native
    relative gap are retained; limit statuses do not certify optimality.
    Improving feasible incumbents are independently scored and published as
    copies, including during native optimisation, when a callback is provided.
    """
    mu, Sigma = _problem(mu, Sigma, lam, k)
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive.")
    if not np.isfinite(mip_rel_gap) or mip_rel_gap < 0:
        raise ValueError("mip_rel_gap must be finite and nonnegative.")
    try:
        from pyscipopt import Model, SCIP_EVENTTYPE, quicksum
    except ImportError as exc:
        raise ImportError(
            "Install vqe-portfolio[scip] or vqe-portfolio[classical] for SCIP."
        ) from exc

    model = Model("exact_k_portfolio")
    try:
        model.hideOutput()
        model.setRealParam("limits/time", float(time_limit))
        model.setRealParam("limits/gap", float(mip_rel_gap))
        model.setRealParam("numerics/feastol", 1e-9)
        model.setIntParam("parallel/maxnthreads", 1)
        x = [model.addVar(name=f"x_{i}", vtype="B") for i in range(len(mu))]
        z = model.addVar(name="objective", vtype="C", lb=None)
        model.addCons(quicksum(x) == k, name="cardinality")
        expression = quicksum(
            float(lam * Sigma[i, i] - mu[i]) * x[i] for i in range(len(mu))
        ) + quicksum(
            float(2 * lam * Sigma[i, j]) * x[i] * x[j]
            for i in range(len(mu))
            for j in range(i + 1, len(mu))
            if lam * Sigma[i, j] != 0
        )
        model.addCons(expression <= z, name="quadratic_objective")
        model.setObjective(z, "minimize")

        def read_candidate():
            solution = model.getBestSol()
            if solution is None:
                return None, False
            raw = np.array([model.getSolVal(solution, variable) for variable in x])
            rounded = np.rint(raw)
            valid = (
                np.all(np.isfinite(raw))
                and np.all(abs(raw - rounded) <= 1e-6)
                and np.all((rounded == 0) | (rounded == 1))
                and rounded.sum() == k
            )
            return (rounded.astype(int), False) if valid else (None, True)

        best_published = np.inf
        callback_errors = []

        def publish(model, event):
            nonlocal best_published
            candidate, _ = read_candidate()
            if candidate is None:
                return
            objective = _objective(candidate, mu, Sigma, lam)
            if objective < best_published:
                best_published = objective
                try:
                    candidate_callback(candidate.copy(), {"source": "scip"})
                except Exception as exc:
                    # Do not propagate Python errors through a native event hook.
                    callback_errors.append(exc)
                    model.interruptSolve()

        if candidate_callback is not None:
            model.attachEventHandlerCallback(publish, [SCIP_EVENTTYPE.BESTSOLFOUND])
        model.optimize()
        if callback_errors:
            raise callback_errors[0]
        if candidate_callback is not None:
            publish(model, None)
            if callback_errors:
                raise callback_errors[0]

        native_status = str(model.getStatus())
        candidate, invalid = read_candidate()

        def finite_bound(value):
            value = float(value)
            return (
                value
                if np.isfinite(value) and not model.isInfinity(abs(value))
                else None
            )

        return ClassicalResult(
            candidate,
            None if candidate is None else _objective(candidate, mu, Sigma, lam),
            "invalid_candidate" if invalid else native_status,
            lower_bound=finite_bound(model.getDualbound()),
            mip_gap=None if candidate is None else finite_bound(model.getGap()),
            message=f"SCIP status={native_status}; time_limit={time_limit}; mip_rel_gap={mip_rel_gap}",
        )
    finally:
        model.freeProb()


def local_search_binary_baseline(
    mu, Sigma, lam, k, *, seed=0, restarts=16, max_swaps=1000, candidate_callback=None
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
        if candidate_callback is not None:
            candidate_callback(x.copy(), {"source": "swap_start"})
        for _ in range(max_swaps):
            inside, outside = np.flatnonzero(x), np.flatnonzero(1 - x)
            if not len(inside) or not len(outside):
                break
            gradient = 2 * Q @ x - mu
            delta = _swap_deltas(Q, gradient, inside, outside)
            a, b = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[a, b] >= -1e-12:
                break
            x[inside[a]], x[outside[b]] = 0, 1
            if candidate_callback is not None:
                candidate_callback(x.copy(), {"source": "swap"})
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


def _swap_deltas(Q, gradient, inside, outside):
    """Objective changes for removing each inside asset and adding each outside."""
    return (
        gradient[outside][None, :]
        - gradient[inside][:, None]
        + np.diag(Q)[inside][:, None]
        + np.diag(Q)[outside][None, :]
        - 2 * Q[np.ix_(inside, outside)]
    )


def _search_starts(mu, Q, k, rng, restarts):
    for restart in range(restarts):
        if restart == 0:
            selected = np.argsort(-mu)[:k]
        elif restart == 1:
            selected = np.argsort(np.diag(Q) - mu)[:k]
        else:
            selected = rng.choice(len(mu), k, replace=False)
        x = np.zeros(len(mu), dtype=int)
        x[selected] = 1
        yield x


def _positive_integer(name, value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")


def simulated_annealing_binary_baseline(
    mu,
    Sigma,
    lam,
    k,
    *,
    seed=0,
    restarts=16,
    max_iterations=1000,
    initial_temperature=None,
    cooling_rate=0.99,
    candidate_callback=None,
) -> ClassicalResult:
    """Seeded Metropolis annealing with exact-K one-for-one swaps.

    Each restart proposes ``max_iterations`` uniformly sampled swaps. The
    temperature is reset at each restart and multiplied by ``cooling_rate``
    after each proposal. By default it starts at the largest absolute swap
    cost at that restart (machine tiny for a flat objective). Explicit initial
    temperatures must be finite and positive. Starts match the swap baseline:
    top returns, diagonal costs, then random subsets.

    Only improving incumbents are published as independent copies. The best
    feasible state across all starts is returned, even if later moves are worse.
    An iteration limit is not a global optimality certificate. NumPy only.
    """
    mu, Sigma = _problem(mu, Sigma, lam, k)
    _positive_integer("restarts", restarts)
    _positive_integer("max_iterations", max_iterations)
    if initial_temperature is not None and (
        not np.isfinite(initial_temperature) or initial_temperature <= 0
    ):
        raise ValueError("initial_temperature must be finite and positive.")
    if not np.isfinite(cooling_rate) or not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be finite and strictly inside (0, 1).")
    rng = np.random.default_rng(seed)
    Q = lam * Sigma
    best, best_obj = None, np.inf
    tiny = np.finfo(float).tiny
    for restart, x in enumerate(_search_starts(mu, Q, k, rng, restarts)):
        obj = _objective(x, mu, Sigma, lam)
        if obj < best_obj:
            best, best_obj = x.copy(), obj
            if candidate_callback is not None:
                candidate_callback(
                    best.copy(), {"source": "anneal_start", "restart": restart}
                )
        if k == 0 or k == len(mu):
            break
        gradient = 2 * Q @ x - mu
        inside, outside = np.flatnonzero(x), np.flatnonzero(1 - x)
        temperature = initial_temperature
        if temperature is None:
            temperature = max(
                float(np.max(abs(_swap_deltas(Q, gradient, inside, outside)))), tiny
            )
        for iteration in range(max_iterations):
            a, b = rng.integers(len(inside)), rng.integers(len(outside))
            i, j = inside[a], outside[b]
            delta = gradient[j] - gradient[i] + Q[i, i] + Q[j, j] - 2 * Q[i, j]
            # Comparing an exponential draw avoids exp overflow/underflow and log(0).
            if delta <= 0 or delta < temperature * rng.exponential():
                x[i], x[j] = 0, 1
                inside[a], outside[b] = j, i
                gradient += 2 * (Q[:, j] - Q[:, i])
                obj = _objective(x, mu, Sigma, lam)
                if obj < best_obj:
                    best, best_obj = x.copy(), obj
                    if candidate_callback is not None:
                        candidate_callback(
                            best.copy(),
                            {
                                "source": "anneal",
                                "restart": restart,
                                "iteration": iteration + 1,
                            },
                        )
            temperature = max(temperature * cooling_rate, tiny)
    return ClassicalResult(
        best,
        best_obj,
        "trivial" if k == 0 or k == len(mu) else "iteration_limit",
        message=f"seed={seed}; restarts={restarts}; max_iterations={max_iterations}; initial_temperature={initial_temperature}; cooling_rate={cooling_rate}",
    )


def tabu_search_binary_baseline(
    mu,
    Sigma,
    lam,
    k,
    *,
    seed=0,
    restarts=16,
    max_iterations=200,
    tabu_tenure=7,
    candidate_callback=None,
) -> ClassicalResult:
    """Multistart best-admissible swap search with reverse-move tabu memory.

    Uphill moves are allowed. After replacing asset i with j, the reverse swap
    j -> i is forbidden for ``tabu_tenure`` subsequent iterations. Aspiration
    overrides that restriction for a new best objective across all restarts.
    Ties follow asset index order. Memory resets at each restart; starts are
    top returns, diagonal costs, then seeded random subsets. A restart stops
    if every move is tabu and none meets aspiration, or at its iteration cap.

    Publishes improving incumbents as independent copies and returns the best
    feasible state, not the final search state. Neither stopping status implies
    global optimality. NumPy only.
    """
    mu, Sigma = _problem(mu, Sigma, lam, k)
    _positive_integer("restarts", restarts)
    _positive_integer("max_iterations", max_iterations)
    _positive_integer("tabu_tenure", tabu_tenure)
    rng = np.random.default_rng(seed)
    Q = lam * Sigma
    best, best_obj = None, np.inf
    capped = False
    for restart, x in enumerate(_search_starts(mu, Q, k, rng, restarts)):
        obj = _objective(x, mu, Sigma, lam)
        if obj < best_obj:
            best, best_obj = x.copy(), obj
            if candidate_callback is not None:
                candidate_callback(
                    best.copy(), {"source": "tabu_start", "restart": restart}
                )
        if k == 0 or k == len(mu):
            break
        gradient = 2 * Q @ x - mu
        tabu_until = np.zeros(Q.shape, dtype=np.int64)
        for iteration in range(max_iterations):
            inside, outside = np.flatnonzero(x), np.flatnonzero(1 - x)
            delta = _swap_deltas(Q, gradient, inside, outside)
            blocked = tabu_until[np.ix_(inside, outside)] > iteration
            aspiration = obj + delta < best_obj - 1e-12
            admissible = np.where(~blocked | aspiration, delta, np.inf)
            a, b = np.unravel_index(np.argmin(admissible), admissible.shape)
            if not np.isfinite(admissible[a, b]):
                break
            i, j = inside[a], outside[b]
            x[i], x[j] = 0, 1
            tabu_until[j, i] = iteration + tabu_tenure + 1
            gradient += 2 * (Q[:, j] - Q[:, i])
            obj = _objective(x, mu, Sigma, lam)
            if obj < best_obj:
                best, best_obj = x.copy(), obj
                if candidate_callback is not None:
                    candidate_callback(
                        best.copy(),
                        {
                            "source": "tabu",
                            "restart": restart,
                            "iteration": iteration + 1,
                        },
                    )
        else:
            capped = True
    status = "iteration_limit" if capped else "tabu_exhausted"
    return ClassicalResult(
        best,
        best_obj,
        "trivial" if k == 0 or k == len(mu) else status,
        message=f"seed={seed}; restarts={restarts}; max_iterations={max_iterations}; tabu_tenure={tabu_tenure}",
    )
