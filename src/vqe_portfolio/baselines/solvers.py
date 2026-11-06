"""Classical portfolio solvers with explicit constraints and solver metadata.

Swap search, annealing, parallel tempering, tabu, memetic GA, ILS, GRASP, VNS,
PGD, and Frank-Wolfe need only NumPy. Optional solvers are imported only when
called. Install ``.[classical]`` for OSQP/CVXPY
and SciPy's HiGHS MILP interface, plus PySCIPOpt's quadratic integer solver.
"""

import json
from dataclasses import dataclass, field

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
    diagnostics: dict = field(default_factory=dict)


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


def _tempering_accept(log_ratio, rng):
    """Metropolis test in log space, without exp overflow or log(0)."""
    return log_ratio >= 0 or -rng.exponential() < log_ratio


def _exchange_log_ratio(cold_energy, hot_energy, cold_temperature, hot_temperature):
    """Log of the swapped/current product-Boltzmann weight ratio.

    Equivalent to (1/T_cold - 1/T_hot)*(E_cold - E_hot). Factoring the
    temperature ratio avoids subtracting overflowing inverse temperatures.
    """
    return ((cold_energy - hot_energy) / cold_temperature) * (
        1 - cold_temperature / hot_temperature
    )


@dataclass
class _TemperingReplica:
    x: np.ndarray
    gradient: np.ndarray
    inside: np.ndarray
    outside: np.ndarray
    energy: float


def parallel_tempering_binary_baseline(
    mu,
    Sigma,
    lam,
    k,
    *,
    seed=0,
    restarts=16,
    max_iterations=1000,
    n_replicas=8,
    temperatures=None,
    exchange_interval=10,
    candidate_callback=None,
) -> ClassicalResult:
    """Replica-exchange Metropolis search for the original exact-K objective.

    Each iteration proposes one uniform selected/unselected swap per temperature
    slot. Every exchange_interval iterations, attempt disjoint neighboring
    exchanges, alternating even/odd edges (always edge 0 for two replicas).
    Exchange complete states and caches, leaving temperatures fixed. Replica
    updates execute serially; parallel tempering names the algorithm, not threads.

    The default fixed geometric ladder spans 0.01*S to S, where S is the largest
    absolute swap cost at the top-return start (1 for a flat start, with a small
    numerical floor). A supplied ladder must have n_replicas distinct, positive,
    finite ascending temperatures. Restarts create independent ladders of states;
    all starts together are top returns, diagonal costs, then random exact-K sets.

    Return the best state from every slot and restart, publishing copied strictly
    improving incumbents. Diagnostics count local proposals/accepts by temperature,
    exchanges by edge, and actual objective evaluations, including initialization.
    This finite heuristic supplies no global optimality bound. NumPy only.
    """
    if k is None:
        raise ValueError("k must be an integer between zero and n_assets.")
    mu, Sigma = _problem(mu, Sigma, lam, k)
    for name, value in (
        ("restarts", restarts),
        ("max_iterations", max_iterations),
        ("n_replicas", n_replicas),
        ("exchange_interval", exchange_interval),
    ):
        _positive_integer(name, value)
    if n_replicas < 2:
        raise ValueError("n_replicas must be at least two.")
    Q = lam * Sigma
    scale_evaluations = 0
    if temperatures is None:
        x = np.zeros(len(mu), dtype=int)
        x[np.argsort(-mu)[:k]] = 1
        scale = 1.0
        if 0 < k < len(mu):
            inside, outside = np.flatnonzero(x), np.flatnonzero(1 - x)
            deltas = _swap_deltas(Q, 2 * Q @ x - mu, inside, outside)
            scale_evaluations = deltas.size
            scale = float(np.max(abs(deltas)))
            if not np.isfinite(scale):
                raise ValueError("Swap costs must be finite to construct temperatures.")
            scale = max(scale, np.finfo(float).tiny * 100) if scale else 1.0
        temperatures = np.geomspace(0.01 * scale, scale, n_replicas)
    if np.iscomplexobj(temperatures):
        raise ValueError("temperatures must be real.")
    ladder = np.array(temperatures, dtype=float, copy=True)
    if (
        ladder.shape != (n_replicas,)
        or not np.all(np.isfinite(ladder))
        or np.any(ladder <= 0)
        or np.any(np.diff(ladder) <= 0)
    ):
        raise ValueError(
            "temperatures must contain n_replicas finite positive ascending values."
        )
    # Python floats avoid NumPy overflow warnings for log ratios at extreme T.
    ladder = ladder.tolist()
    rng = np.random.default_rng(seed)
    starts = iter(_search_starts(mu, Q, k, rng, restarts * n_replicas))
    local_proposals = np.zeros(n_replicas, dtype=int)
    local_accepts = np.zeros(n_replicas, dtype=int)
    exchange_attempts = np.zeros(n_replicas - 1, dtype=int)
    exchange_accepts = np.zeros(n_replicas - 1, dtype=int)
    objective_evaluations = iterations = completed_restarts = 0
    best, best_obj = None, np.inf

    def diagnostics():
        def rates(accepts, attempts):
            return [int(a) / int(b) if b else None for a, b in zip(accepts, attempts)]

        return {
            "temperatures": list(ladder),
            "n_replicas": int(n_replicas),
            "restarts": int(restarts),
            "max_iterations": int(max_iterations),
            "exchange_interval": int(exchange_interval),
            "completed_restarts": completed_restarts,
            "iterations": iterations,
            "local_proposals": int(local_proposals.sum()),
            "local_accepts": int(local_accepts.sum()),
            "local_proposals_by_temperature": local_proposals.tolist(),
            "local_accepts_by_temperature": local_accepts.tolist(),
            "local_acceptance_rates": rates(local_accepts, local_proposals),
            "exchange_attempts": int(exchange_attempts.sum()),
            "exchange_accepts": int(exchange_accepts.sum()),
            "exchange_attempts_by_edge": exchange_attempts.tolist(),
            "exchange_accepts_by_edge": exchange_accepts.tolist(),
            "exchange_acceptance_rates": rates(exchange_accepts, exchange_attempts),
            "objective_evaluations": objective_evaluations,
            "temperature_scale_swap_evaluations": int(scale_evaluations),
        }

    def consider(x, energy, restart, iteration, slot):
        nonlocal best, best_obj
        if energy < best_obj:
            best, best_obj = x.copy(), energy
            if candidate_callback is not None:
                candidate_callback(
                    best.copy(),
                    {
                        "source": "parallel-tempering",
                        "restart": restart,
                        "iteration": iteration,
                        "temperature_slot": slot,
                        "solver_diagnostics": diagnostics(),
                    },
                )

    for restart in range(restarts):
        replicas = []
        for slot in range(n_replicas):
            x = next(starts)
            energy = _objective(x, mu, Sigma, lam)
            objective_evaluations += 1
            if not np.isfinite(energy):
                raise ValueError("Portfolio energies must be finite.")
            consider(x, energy, restart, 0, slot)
            replicas.append(
                _TemperingReplica(
                    x, 2 * Q @ x - mu, np.flatnonzero(x), np.flatnonzero(1 - x), energy
                )
            )
            if k == 0 or k == len(mu):
                break
        if k == 0 or k == len(mu):
            break
        for iteration in range(1, max_iterations + 1):
            for slot, state in enumerate(replicas):
                a, b = rng.integers(k), rng.integers(len(mu) - k)
                i, j = state.inside[a], state.outside[b]
                delta = float(
                    state.gradient[j]
                    - state.gradient[i]
                    + Q[i, i]
                    + Q[j, j]
                    - 2 * Q[i, j]
                )
                if not np.isfinite(delta):
                    raise ValueError("Swap costs must be finite.")
                local_proposals[slot] += 1
                if _tempering_accept(-delta / ladder[slot], rng):
                    local_accepts[slot] += 1
                    state.x[i], state.x[j] = 0, 1
                    state.inside[a], state.outside[b] = j, i
                    state.gradient += 2 * (Q[:, j] - Q[:, i])
                    state.energy = _objective(state.x, mu, Sigma, lam)
                    objective_evaluations += 1
                    if not np.isfinite(state.energy):
                        raise ValueError("Portfolio energies must be finite.")
                    consider(state.x, state.energy, restart, iteration, slot)
            iterations += 1
            if iteration % exchange_interval == 0:
                round_index = iteration // exchange_interval - 1
                parity = round_index % 2 if n_replicas > 2 else 0
                for slot in range(parity, n_replicas - 1, 2):
                    exchange_attempts[slot] += 1
                    log_ratio = _exchange_log_ratio(
                        replicas[slot].energy,
                        replicas[slot + 1].energy,
                        ladder[slot],
                        ladder[slot + 1],
                    )
                    if _tempering_accept(log_ratio, rng):
                        exchange_accepts[slot] += 1
                        replicas[slot], replicas[slot + 1] = (
                            replicas[slot + 1],
                            replicas[slot],
                        )
        completed_restarts += 1
    details = diagnostics()
    return ClassicalResult(
        best,
        best_obj,
        "trivial" if k == 0 or k == len(mu) else "iteration_limit",
        message=f"seed={seed}; diagnostics={json.dumps(details, sort_keys=True)}; no certified bound",
        diagnostics=details,
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


class _BinaryLocalSearch:
    """Shared swap refinement and independently scored, copied incumbents."""

    def __init__(self, mu, Sigma, lam, k, max_swaps, callback):
        self.mu, self.Sigma = _problem(mu, Sigma, lam, k)
        _positive_integer("max_swaps", max_swaps)
        self.lam, self.Q = lam, lam * self.Sigma
        self.max_swaps, self.callback = max_swaps, callback
        self.best, self.best_obj = None, np.inf

    def consider(self, x, source, **details):
        score = _objective(x, self.mu, self.Sigma, self.lam)
        if score < self.best_obj:
            self.best, self.best_obj = x.copy(), score
            if self.callback is not None:
                self.callback(self.best.copy(), {"source": source, **details})
        return score

    def refine(self, x, source, **details):
        x = x.copy()
        self.consider(x, source, stage="start", **details)
        for _ in range(self.max_swaps):
            inside, outside = np.flatnonzero(x), np.flatnonzero(1 - x)
            if not len(inside) or not len(outside):
                break
            delta = _swap_deltas(self.Q, 2 * self.Q @ x - self.mu, inside, outside)
            a, b = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[a, b] >= -1e-12:
                break
            x[inside[a]], x[outside[b]] = 0, 1
            self.consider(x, source, stage="swap", **details)
        return x

    def result(self, status, message):
        return ClassicalResult(self.best.copy(), self.best_obj, status, message=message)


def relaxation_rounding_binary_baseline(
    mu,
    Sigma,
    lam,
    k,
    *,
    max_swaps=1000,
    qp_max_iter=100000,
    candidate_callback=None,
) -> ClassicalResult:
    """Convex box relaxation, deterministic top-K rounding, then swap refinement.

    Solve the original quadratic objective on 0 <= z <= 1, sum(z) == K using
    CVXPY/OSQP. This retains quadratic diagonal terms and the binary track's raw
    selection scale. A PSD covariance is required. Round the largest K entries
    (asset-index order for exact ties), then perform up to ``max_swaps`` improving
    exchanges. Only complete binary incumbents are published as copied arrays.

    The numerical relaxation status and recomputed objective are diagnostic
    notes, not a certified bound. ``rounded_and_refined`` is a heuristic status,
    not integer optimality. Bad or missing relaxation output is returned visibly
    without substituting another solver. Native solver exceptions propagate.
    """
    search = _BinaryLocalSearch(mu, Sigma, lam, k, max_swaps, candidate_callback)
    _positive_integer("qp_max_iter", qp_max_iter)
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "Install vqe-portfolio[classical] or vqe-portfolio[markowitz] for QP rounding."
        ) from exc

    z = cp.Variable(len(search.mu))
    relaxation = cp.Problem(
        cp.Minimize(lam * cp.quad_form(z, search.Sigma) - search.mu @ z),
        [z >= 0, z <= 1, cp.sum(z) == k],
    )
    if not relaxation.is_dcp():
        raise ValueError("The QP relaxation requires a PSD covariance.")
    if k == 0 or k == len(search.mu):
        x = np.zeros(len(search.mu), dtype=int)
        x[:k] = 1
        search.consider(x, "qp-round", stage="trivial")
        return search.result(
            "trivial", "Single feasible selection; no relaxation solve required"
        )

    relaxation.solve(
        solver="OSQP",
        eps_abs=1e-9,
        eps_rel=1e-9,
        max_iter=int(qp_max_iter),
        warm_start=False,
    )
    native_status = str(relaxation.status)
    notes = f"relaxation_status={native_status}; qp_max_iter={qp_max_iter}; max_swaps={max_swaps}; no certified lower bound"
    if z.value is None:
        return ClassicalResult(None, None, native_status, message=notes)
    raw = np.asarray(z.value)
    if (
        raw.shape != search.mu.shape
        or np.iscomplexobj(raw)
        or not np.all(np.isfinite(raw))
        or raw.min() < -1e-8
        or raw.max() > 1 + 1e-8
        or abs(raw.sum() - k) > 1e-8
    ):
        return ClassicalResult(None, None, "invalid_relaxation", message=notes)
    relaxed_objective = _objective(raw, search.mu, search.Sigma, lam)
    if not np.isfinite(relaxed_objective):
        return ClassicalResult(None, None, "invalid_relaxation", message=notes)
    x = np.zeros(len(raw), dtype=int)
    x[np.argsort(-np.clip(raw, 0, 1), kind="stable")[:k]] = 1
    search.refine(x, "qp-round", relaxation_status=native_status)
    return search.result(
        "rounded_and_refined",
        f"{notes}; relaxation_objective={relaxed_objective}",
    )


def _perturb_selection(x, rng, swaps):
    """Exchange distinct assets simultaneously, clamping to the available sets."""
    x = x.copy()
    inside, outside = np.flatnonzero(x), np.flatnonzero(1 - x)
    count = min(swaps, len(inside), len(outside))
    x[rng.choice(inside, count, replace=False)] = 0
    x[rng.choice(outside, count, replace=False)] = 1
    return x


def _subset_crossover(first, second, rng):
    """Keep common assets and fill the remaining cardinality from the union."""
    child = first & second
    pool = np.flatnonzero(first ^ second)
    count = int(first.sum() - child.sum())
    child[rng.choice(pool, count, replace=False)] = 1
    return child


def memetic_genetic_binary_baseline(
    mu,
    Sigma,
    lam,
    k,
    *,
    seed=0,
    restarts=16,
    population_size=16,
    generations=25,
    mutation_rate=0.2,
    tournament_size=3,
    max_swaps=1000,
    candidate_callback=None,
) -> ClassicalResult:
    """Exact-K genetic search with swap refinement, elitism, and immigration.

    Each independent population starts with top returns, diagonal costs, then
    random subsets. Every initial member and offspring receives up to
    ``max_swaps`` best-improvement swaps. Parents are selected by tournaments
    sampled with replacement. Crossover retains common assets and samples the
    remaining assets from the parents' union. Mutation performs one exchange
    with probability ``mutation_rate``. Each generation keeps its best member
    and, for population sizes >= 3, includes one random immigrant.

    Duplicates are allowed so populations can exceed the feasible-space size
    without retry loops. Only improving feasible incumbents are published as
    copies. Returns the best portfolio across populations; no optimality bound.
    """
    search = _BinaryLocalSearch(mu, Sigma, lam, k, max_swaps, candidate_callback)
    for name, value in [
        ("restarts", restarts),
        ("population_size", population_size),
        ("generations", generations),
        ("tournament_size", tournament_size),
    ]:
        _positive_integer(name, value)
    if population_size < 2:
        raise ValueError("population_size must be at least two.")
    if not np.isfinite(mutation_rate) or not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be finite and in [0, 1].")
    rng = np.random.default_rng(seed)
    trivial = k == 0 or k == len(search.mu)
    for restart in range(restarts):
        population, scores = [], []
        for member, x in enumerate(
            _search_starts(search.mu, search.Q, k, rng, population_size)
        ):
            x = search.refine(
                x, "memetic", restart=restart, generation=0, member=member
            )
            population.append(x)
            scores.append(search.consider(x, "memetic", restart=restart, generation=0))
            if trivial:
                break
        if trivial:
            break
        for generation in range(1, generations + 1):
            elite = int(np.argmin(scores))
            offspring, next_scores = [population[elite].copy()], [scores[elite]]

            def parent():
                contestants = rng.integers(len(population), size=tournament_size)
                return population[min(contestants, key=lambda i: scores[i])]

            for member in range(1, population_size):
                if population_size >= 3 and member == population_size - 1:
                    child = np.zeros(len(search.mu), dtype=int)
                    child[rng.choice(len(child), k, replace=False)] = 1
                else:
                    child = _subset_crossover(parent(), parent(), rng)
                    if rng.random() < mutation_rate:
                        child = _perturb_selection(child, rng, 1)
                child = search.refine(
                    child,
                    "memetic",
                    restart=restart,
                    generation=generation,
                    member=member,
                )
                offspring.append(child)
                next_scores.append(
                    search.consider(
                        child, "memetic", restart=restart, generation=generation
                    )
                )
            population, scores = offspring, next_scores
    return search.result(
        "trivial" if trivial else "generation_limit",
        f"seed={seed}; restarts={restarts}; population_size={population_size}; generations={generations}; mutation_rate={mutation_rate}; tournament_size={tournament_size}; max_swaps={max_swaps}",
    )


def iterated_local_search_binary_baseline(
    mu,
    Sigma,
    lam,
    k,
    *,
    seed=0,
    restarts=16,
    max_iterations=100,
    perturbation_size=2,
    max_swaps=1000,
    candidate_callback=None,
) -> ClassicalResult:
    """Perturb a locally refined portfolio, refine again, and accept improvements.

    Starts are top returns, diagonal costs, then seeded random subsets. Each
    iteration simultaneously replaces ``perturbation_size`` distinct selected
    assets with unselected assets, clamped to min(K, n-K), then performs up to
    ``max_swaps`` improving swaps. Strictly better refined portfolios replace
    the current state; rejected perturbations never overwrite the incumbent.
    Publishes improving feasible copies and returns the best across restarts.
    Iteration limits imply no global optimality guarantee.
    """
    search = _BinaryLocalSearch(mu, Sigma, lam, k, max_swaps, candidate_callback)
    for name, value in [
        ("restarts", restarts),
        ("max_iterations", max_iterations),
        ("perturbation_size", perturbation_size),
    ]:
        _positive_integer(name, value)
    rng = np.random.default_rng(seed)
    trivial = k == 0 or k == len(search.mu)
    for restart, x in enumerate(_search_starts(search.mu, search.Q, k, rng, restarts)):
        current = search.refine(x, "ils", restart=restart, iteration=0)
        current_obj = search.consider(current, "ils", restart=restart, iteration=0)
        if trivial:
            break
        for iteration in range(1, max_iterations + 1):
            x = _perturb_selection(current, rng, perturbation_size)
            refined = search.refine(x, "ils", restart=restart, iteration=iteration)
            score = search.consider(
                refined, "ils", restart=restart, iteration=iteration
            )
            if score < current_obj - 1e-12:
                current, current_obj = refined, score
    return search.result(
        "trivial" if trivial else "iteration_limit",
        f"seed={seed}; restarts={restarts}; max_iterations={max_iterations}; perturbation_size={perturbation_size}; max_swaps={max_swaps}",
    )


def grasp_binary_baseline(
    mu,
    Sigma,
    lam,
    k,
    *,
    seed=0,
    restarts=16,
    rcl_alpha=0.3,
    max_swaps=1000,
    candidate_callback=None,
) -> ClassicalResult:
    """Greedy randomised construction followed by best-improvement swaps.

    Build each exact-K subset from empty using the full objective's marginal
    insertion costs. Sample uniformly from the restricted candidate list of
    costs <= min_cost + rcl_alpha * (max_cost - min_cost). Alpha zero chooses
    minimum-cost assets (random ties); alpha one permits every remaining asset.
    Construction never publishes incomplete selections. Refine each completed
    subset with up to ``max_swaps`` swaps and retain the best across restarts.
    Published incumbents are copied and feasible; no global bound is implied.
    """
    search = _BinaryLocalSearch(mu, Sigma, lam, k, max_swaps, candidate_callback)
    _positive_integer("restarts", restarts)
    if not np.isfinite(rcl_alpha) or not 0 <= rcl_alpha <= 1:
        raise ValueError("rcl_alpha must be finite and in [0, 1].")
    rng = np.random.default_rng(seed)
    trivial = k == 0 or k == len(search.mu)
    for restart in range(restarts):
        x = np.zeros(len(search.mu), dtype=int)
        gradient = -search.mu.copy()
        if trivial:
            x[:k] = 1
        for _ in range(0 if trivial else k):
            remaining = np.flatnonzero(1 - x)
            costs = gradient[remaining] + np.diag(search.Q)[remaining]
            threshold = (
                costs.max()
                if rcl_alpha == 1
                else costs.min() + rcl_alpha * (costs.max() - costs.min())
            )
            chosen = rng.choice(remaining[costs <= threshold])
            x[chosen] = 1
            gradient += 2 * search.Q[:, chosen]
        search.refine(x, "grasp", restart=restart)
        if trivial:
            break
    return search.result(
        "trivial" if trivial else "restart_limit",
        f"seed={seed}; restarts={restarts}; rcl_alpha={rcl_alpha}; max_swaps={max_swaps}",
    )


def variable_neighborhood_search_binary_baseline(
    mu,
    Sigma,
    lam,
    k,
    *,
    seed=0,
    restarts=16,
    max_iterations=100,
    max_neighborhood=3,
    max_swaps=1000,
    candidate_callback=None,
) -> ClassicalResult:
    """Basic VNS: shake increasing exact-exchange neighbourhoods and refine.

    Starts match swap search: top returns, diagonal costs, then random subsets.
    At neighbourhood q, simultaneously exchange q distinct selected assets for
    q distinct unselected assets, then apply best-improvement single swaps.
    Strict improvement replaces the current portfolio and resets q to one;
    otherwise advance q, wrapping to one after min(max_neighborhood, K, n-K).
    Each restart gets at most max_iterations shake-and-refine attempts.

    Only improving feasible incumbents are published as independent copies.
    Neither the iteration cap nor local refinement implies global optimality.
    """
    search = _BinaryLocalSearch(mu, Sigma, lam, k, max_swaps, candidate_callback)
    for name, value in [
        ("restarts", restarts),
        ("max_iterations", max_iterations),
        ("max_neighborhood", max_neighborhood),
    ]:
        _positive_integer(name, value)
    rng = np.random.default_rng(seed)
    trivial = k == 0 or k == len(search.mu)
    limit = min(max_neighborhood, k, len(search.mu) - k)
    for restart, x in enumerate(_search_starts(search.mu, search.Q, k, rng, restarts)):
        current = search.refine(x, "vns", restart=restart, iteration=0, neighborhood=0)
        current_obj = search.consider(current, "vns", restart=restart, iteration=0)
        if trivial:
            break
        neighborhood = 1
        for iteration in range(1, max_iterations + 1):
            shaken = _perturb_selection(current, rng, neighborhood)
            refined = search.refine(
                shaken,
                "vns",
                restart=restart,
                iteration=iteration,
                neighborhood=neighborhood,
            )
            score = search.consider(
                refined, "vns", restart=restart, iteration=iteration
            )
            if score < current_obj - 1e-12:
                current, current_obj, neighborhood = refined, score, 1
            else:
                neighborhood = neighborhood + 1 if neighborhood < limit else 1
    return search.result(
        "trivial" if trivial else "iteration_limit",
        f"seed={seed}; restarts={restarts}; max_iterations={max_iterations}; max_neighborhood={max_neighborhood}; resolved_max_neighborhood={limit}; max_swaps={max_swaps}",
    )


def _project_simplex(vector):
    """Euclidean projection onto nonnegative weights summing to one."""
    # Translation leaves the projection unchanged and avoids losing the unit
    # budget when all entries share a large constant offset.
    shifted = vector - np.max(vector)
    ordered = np.sort(shifted)[::-1]
    cumulative = np.cumsum(ordered) - 1
    sizes = np.arange(1, len(vector) + 1)
    rho = np.flatnonzero(ordered - cumulative / sizes > 0)[-1]
    weights = np.maximum(shifted - cumulative[rho] / (rho + 1), 0)
    weights /= weights.sum()
    return weights


def projected_gradient_descent_baseline(
    mu,
    Sigma,
    lam,
    *,
    max_iterations=10000,
    tolerance=1e-8,
    stepsize=None,
    candidate_callback=None,
) -> ClassicalResult:
    """NumPy projected gradient descent for the long-only convex simplex QP.

    Start at equal weights. Use 1/L for L=2*lam*lambda_max(Sigma), unless an
    initial stepsize is supplied, and backtrack at most 60 times per iteration
    until quadratic majorisation and numerical non-increase hold. Project each
    trial onto the simplex. Zero-curvature objectives use their exact linear
    simplex solution. Covariance eigenvalues below -1e-10 are rejected.

    Stop when the returned best portfolio's numerical first-order gap is within
    tolerance. The gap is grad(f)@w - min(grad(f)), not a certified bound. Keep
    and publish only strictly improving feasible copies, including the initial
    portfolio. Iteration and line-search limits can retain an incumbent.
    """
    mu, Sigma = _problem(mu, Sigma, lam)
    _positive_integer("max_iterations", max_iterations)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if stepsize is not None and (not np.isfinite(stepsize) or stepsize <= 0):
        raise ValueError("stepsize must be finite and positive.")
    eigenvalues = np.linalg.eigvalsh(Sigma)
    if eigenvalues[0] < -1e-10:
        raise ValueError("Projected gradient descent requires a PSD covariance.")
    Q = lam * Sigma
    lipschitz = 2 * lam * max(0.0, float(eigenvalues[-1]))
    step = (
        float(stepsize)
        if stepsize is not None
        else (1 / lipschitz if lipschitz > 0 else 1.0)
    )
    current = np.full(len(mu), 1 / len(mu))
    current /= current.sum()
    best, best_obj = None, np.inf

    def consider(w, iteration):
        nonlocal best, best_obj
        score = _objective(w, mu, Sigma, lam)
        if score < best_obj:
            best, best_obj = w.copy(), score
            if candidate_callback is not None:
                candidate_callback(
                    best.copy(), {"source": "pgd", "iteration": iteration}
                )
        return score

    objective = consider(current, 0)
    if lipschitz == 0:
        vertex = np.zeros(len(mu))
        vertex[np.argmax(mu)] = 1
        consider(vertex, 1)
        status, iteration = "converged", 1
    else:
        status = "iteration_limit"
        for iteration in range(max_iterations + 1):
            gradient = 2 * Q @ current - mu
            gap = float((gradient - gradient.min()) @ current)
            if gap <= tolerance:
                status = "converged"
                break
            if iteration == max_iterations:
                break
            for _ in range(60):
                with np.errstate(over="ignore", invalid="ignore"):
                    trial = current - step * gradient
                if np.all(np.isfinite(trial)):
                    candidate = _project_simplex(trial)
                    candidate_obj = _objective(candidate, mu, Sigma, lam)
                    delta = candidate - current
                    model = objective + gradient @ delta + (delta @ delta) / (2 * step)
                    slack = 1e-14 * max(1.0, abs(objective))
                    if (
                        np.isfinite(candidate_obj)
                        and candidate_obj <= min(model, objective) + slack
                    ):
                        break
                step *= 0.5
                if step == 0:
                    break
            else:
                status = "line_search_limit"
                break
            if step == 0:
                status = "line_search_limit"
                break
            current, objective = candidate, candidate_obj
            consider(current, iteration + 1)
    gradient = 2 * Q @ best - mu
    best_gap = float((gradient - gradient.min()) @ best)
    if status == "converged" and best_gap > tolerance:
        status = "numerical_limit"
    return ClassicalResult(
        best,
        best_obj,
        status,
        message=f"iterations={iteration}; max_iterations={max_iterations}; tolerance={tolerance}; initial_stepsize={stepsize}; final_stepsize={step}; first_order_gap={best_gap}; no certified bound",
    )


def frank_wolfe_baseline(
    mu,
    Sigma,
    lam,
    *,
    max_iterations=10000,
    tolerance=1e-8,
    candidate_callback=None,
) -> ClassicalResult:
    """Away-steps Frank-Wolfe for the long-only convex simplex QP.

    Implements Algorithm 1 of Lacoste-Julien and Jaggi (2015): compare the
    forward and away directional gaps, then minimize the quadratic along the
    selected direction within its feasible step bound. Simplex coordinates are
    the active vertices' coefficients. Start at the largest-return vertex;
    ties in vertex selection use the lowest index. No projection is used.

    Covariance eigenvalues below -1e-10 are rejected. Stop on the numerical
    Frank-Wolfe gap grad(f)@w - min(grad(f)), not an independently certified
    bound. Retain and publish copied strictly improving feasible incumbents.
    Callbacks include the chosen atom, step type, step size, and feasible bound.
    """
    mu, Sigma = _problem(mu, Sigma, lam)
    _positive_integer("max_iterations", max_iterations)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if np.linalg.eigvalsh(Sigma)[0] < -1e-10:
        raise ValueError("Frank-Wolfe requires a PSD covariance.")
    Q = lam * Sigma
    current = np.zeros(len(mu))
    current[np.argmax(mu)] = 1
    best, best_obj = current.copy(), _objective(current, mu, Sigma, lam)
    objective = best_obj
    forward_steps = away_steps = drop_steps = 0
    if candidate_callback is not None:
        candidate_callback(
            best.copy(),
            {"source": "frank-wolfe", "iteration": 0, "step_type": "initial"},
        )

    status = "iteration_limit"
    for iteration in range(max_iterations + 1):
        gradient = 2 * Q @ current - mu
        if not np.all(np.isfinite(gradient)):
            status = "numerical_limit"
            break
        forward_atom = int(np.argmin(gradient))
        # Center before averaging to reduce cancellation from constant offsets.
        centered = gradient - gradient[forward_atom]
        forward_gap = float(centered @ current)
        if forward_gap <= tolerance:
            status = "converged"
            break
        if iteration == max_iterations:
            break
        active = np.flatnonzero(current > 0)
        away_atom = int(active[np.argmax(centered[active])])
        away_gap = float(centered[away_atom] - forward_gap)
        if len(active) > 1 and away_gap > forward_gap:
            atom, step_type, directional_gap = away_atom, "away", away_gap
            direction = current.copy()
            direction[atom] -= 1
            max_step = float(current[atom] / (1 - current[atom]))
        else:
            atom, step_type, directional_gap = forward_atom, "forward", forward_gap
            direction = -current
            direction[atom] += 1
            max_step = 1.0
        curvature = float(direction @ Q @ direction)
        if not np.isfinite(curvature):
            status = "numerical_limit"
            break
        step = (
            min(max_step, directional_gap / (2 * curvature))
            if curvature > 0
            else max_step
        )
        if step_type == "forward":
            candidate = (1 - step) * current
            candidate[atom] += step
        else:
            candidate = (1 + step) * current
            # A maximal away step removes the atom exactly, without leaving
            # cancellation-sized coefficients in the active set.
            candidate[atom] = (
                0.0 if step == max_step else current[atom] - step * (1 - current[atom])
            )
        # Correct only floating-point drift in the convex coefficients.
        candidate /= candidate.sum()
        candidate_obj = _objective(candidate, mu, Sigma, lam)
        slack = 1e-14 * max(1.0, abs(objective))
        if (
            not np.all(np.isfinite(candidate))
            or np.any(candidate < 0)
            or not np.isfinite(candidate_obj)
            or candidate_obj > objective + slack
            or np.array_equal(candidate, current)
        ):
            status = "numerical_limit"
            break
        forward_steps += step_type == "forward"
        away_steps += step_type == "away"
        drop_steps += step_type == "away" and step == max_step
        current, objective = candidate, candidate_obj
        if objective < best_obj:
            best, best_obj = current.copy(), objective
            if candidate_callback is not None:
                candidate_callback(
                    best.copy(),
                    {
                        "source": "frank-wolfe",
                        "iteration": iteration + 1,
                        "step_type": step_type,
                        "atom": atom,
                        "step_size": step,
                        "max_step_size": max_step,
                    },
                )

    gradient = 2 * Q @ best - mu
    best_gap = float((gradient - gradient.min()) @ best)
    if status == "converged" and best_gap > tolerance:
        status = "numerical_limit"
    return ClassicalResult(
        best,
        best_obj,
        status,
        message=f"iterations={forward_steps + away_steps}; max_iterations={max_iterations}; tolerance={tolerance}; forward_steps={forward_steps}; away_steps={away_steps}; drop_steps={drop_steps}; active_set_size={np.count_nonzero(best)}; first_order_gap={best_gap}; no certified bound",
    )
