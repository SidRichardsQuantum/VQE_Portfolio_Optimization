from types import SimpleNamespace

import numpy as np
import pytest

from vqe_portfolio.baselines import (
    convex_markowitz_baseline,
    local_search_binary_baseline,
    mixed_integer_binary_baseline,
    scip_binary_baseline,
    simulated_annealing_binary_baseline,
    tabu_search_binary_baseline,
)
from vqe_portfolio.comparison import (
    exact_long_only_markowitz_baseline,
    exhaustive_binary_baseline,
    top_return_baseline,
)


def problem(seed, n=6):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, n))
    return rng.normal(size=n), factors @ factors.T / n


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", [0, 1, 3, 6])
def test_milp_matches_enumeration(seed, k):
    pytest.importorskip("scipy.optimize")
    mu, sigma = problem(seed)
    result = mixed_integer_binary_baseline(mu, sigma, 0.7, k)
    _, expected = exhaustive_binary_baseline(mu, sigma, 0.7, 100, k)
    assert result.status == "optimal"
    assert result.candidate.sum() == k
    assert result.objective == pytest.approx(expected, abs=1e-7)
    assert result.lower_bound == pytest.approx(expected, abs=1e-7)
    assert result.mip_gap == pytest.approx(0, abs=1e-7)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lam", [0, 0.7, 5])
def test_qp_matches_enumeration(seed, lam):
    pytest.importorskip("cvxpy")
    pytest.importorskip("osqp")
    mu, sigma = problem(seed)
    result = convex_markowitz_baseline(mu, sigma, lam)
    _, expected = exact_long_only_markowitz_baseline(mu, sigma, lam)
    assert result.status == "optimal"
    assert result.objective == pytest.approx(expected, abs=1e-7)
    assert np.all(result.candidate >= 0)
    assert result.candidate.sum() == pytest.approx(1)
    assert result.lower_bound is None


@pytest.mark.parametrize("k", [0, 1, 3, 6])
def test_local_search_feasibility_reproducibility_and_local_optimum(k):
    mu, sigma = problem(40)
    result = local_search_binary_baseline(mu, sigma, 0.7, k, seed=15)
    again = local_search_binary_baseline(mu, sigma, 0.7, k, seed=15)
    np.testing.assert_array_equal(result.candidate, again.candidate)
    assert result.candidate.sum() == k
    _, exact = exhaustive_binary_baseline(mu, sigma, 0.7, 100, k)
    assert result.objective >= exact - 1e-10
    # k=0 is intentionally excluded: the legacy top-return helper uses [-k:].
    if k:
        _, top = top_return_baseline(mu, sigma, 0.7, 100, k)
        assert result.objective <= top + 1e-10
    for i in np.flatnonzero(result.candidate):
        for j in np.flatnonzero(1 - result.candidate):
            neighbour = result.candidate.copy()
            neighbour[i], neighbour[j] = 0, 1
            obj = 0.7 * neighbour @ sigma @ neighbour - mu @ neighbour
            assert obj >= result.objective - 1e-10


def test_qp_singular_covariance():
    pytest.importorskip("cvxpy")
    result = convex_markowitz_baseline(np.array([0.1, 0.2, 0.3]), np.zeros((3, 3)), 4)
    assert result.objective == pytest.approx(-0.3, abs=1e-8)


@pytest.mark.parametrize("incumbent", [None, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
def test_milp_time_limit_retains_status_and_bound(monkeypatch, incumbent):
    optimize = pytest.importorskip("scipy.optimize")
    monkeypatch.setattr(
        optimize,
        "milp",
        lambda *a, **kw: SimpleNamespace(
            status=1,
            x=None if incumbent is None else np.array(incumbent),
            mip_dual_bound=-2.0,
            mip_gap=None,
            message="Time limit reached",
        ),
    )
    result = mixed_integer_binary_baseline(np.array([1.0, 2.0, 3.0]), np.eye(3), 1, 1)
    assert result.status == "limit"
    assert result.lower_bound == -2
    assert (result.candidate is None) == (incumbent is None)


@pytest.mark.parametrize("k", [-1, 7, 1.5])
def test_invalid_cardinality(k):
    mu, sigma = problem(1)
    for solve in [
        mixed_integer_binary_baseline,
        scip_binary_baseline,
        local_search_binary_baseline,
        simulated_annealing_binary_baseline,
        tabu_search_binary_baseline,
    ]:
        with pytest.raises(ValueError, match="k must"):
            solve(mu, sigma, 1, k)


def test_nonconvex_qp_rejected():
    pytest.importorskip("cvxpy")
    with pytest.raises(ValueError, match="PSD"):
        convex_markowitz_baseline(np.ones(2), -np.eye(2), 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_limit": 0},
        {"time_limit": -1},
        {"time_limit": float("nan")},
        {"time_limit": float("inf")},
        {"mip_rel_gap": -1},
        {"mip_rel_gap": float("nan")},
        {"mip_rel_gap": float("inf")},
    ],
)
def test_scip_rejects_invalid_limits_before_importing(kwargs):
    mu, sigma = problem(1)
    with pytest.raises(ValueError):
        scip_binary_baseline(mu, sigma, 1, 3, **kwargs)


def test_scip_missing_dependency_has_install_hint(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pyscipopt":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"vqe-portfolio\[scip\]"):
        scip_binary_baseline(np.ones(3), np.eye(3), 1, 2)


SEARCH_SOLVERS = [simulated_annealing_binary_baseline, tabu_search_binary_baseline]


@pytest.mark.parametrize("solve", SEARCH_SOLVERS)
@pytest.mark.parametrize("seed", [0, 15, 40])
@pytest.mark.parametrize("k", [0, 1, 3, 6])
def test_stochastic_search_feasibility_reproducibility_and_incumbents(solve, seed, k):
    from vqe_portfolio.problem import PortfolioProblem

    mu, sigma = problem(seed)
    prepared = PortfolioProblem(mu, sigma, 0.7, "binary_qubo", k)
    before_mu, before_sigma = mu.copy(), sigma.copy()
    published = []

    def publish(candidate, details):
        published.append(candidate.copy())
        # A consumer cannot mutate either the search state or its retained best.
        candidate[:] = -10

    kwargs = dict(seed=seed, restarts=3, max_iterations=100)
    result = solve(mu, sigma, 0.7, k, candidate_callback=publish, **kwargs)
    again = solve(prepared.mu, prepared.Sigma, 0.7, k, **kwargs)
    np.testing.assert_array_equal(result.candidate, again.candidate)
    np.testing.assert_array_equal(mu, before_mu)
    np.testing.assert_array_equal(sigma, before_sigma)
    assert result.objective == again.objective
    scores = [prepared.evaluate(x) for x in published]
    assert all(score.feasible for score in scores)
    assert all(a.objective > b.objective for a, b in zip(scores, scores[1:]))
    np.testing.assert_array_equal(result.candidate, published[-1])
    assert result.objective == pytest.approx(
        prepared.evaluate(result.candidate).objective
    )
    _, exact = exhaustive_binary_baseline(mu, sigma, 0.7, 0, k)
    assert result.objective >= exact - 1e-10
    start = np.zeros(len(mu))
    start[np.argsort(-mu)[:k]] = 1
    assert result.objective <= prepared.evaluate(start).objective + 1e-10
    assert result.lower_bound is None and result.mip_gap is None
    assert result.status == ("trivial" if k in {0, 6} else "iteration_limit")


@pytest.fixture
def local_minimum_problem():
    # Two correlated pairs. Either single swap out of the first pair is worse,
    # but selecting the entire second pair is globally better. Covariance is PSD.
    mu = np.array([1.0, 1.0, 0.1, 0.1])
    sigma = np.array(
        [[5, -3, 0, 0], [-3, 5, 0, 0], [0, 0, 5.5, -5.4], [0, 0, -5.4, 5.5]]
    )
    return mu, sigma


@pytest.mark.parametrize("solve", SEARCH_SOLVERS)
def test_search_escapes_swap_local_minimum(solve, local_minimum_problem):
    mu, sigma = local_minimum_problem
    greedy = local_search_binary_baseline(mu, sigma, 1, 2, restarts=2)
    result = solve(mu, sigma, 1, 2, seed=0, restarts=1, max_iterations=1000)
    _, exact = exhaustive_binary_baseline(mu, sigma, 1, 0, 2)
    assert result.objective < greedy.objective
    assert result.objective == pytest.approx(exact)


def test_annealing_temperature_controls_uphill_acceptance(local_minimum_problem):
    mu, sigma = local_minimum_problem
    cold = simulated_annealing_binary_baseline(
        mu, sigma, 1, 2, restarts=1, max_iterations=1000, initial_temperature=1e-12
    )
    np.testing.assert_array_equal(cold.candidate, [1, 1, 0, 0])
    warm = simulated_annealing_binary_baseline(
        mu, sigma, 1, 2, restarts=1, max_iterations=1000, initial_temperature=10
    )
    assert warm.objective < cold.objective


def test_tabu_blocks_reverse_moves_and_retains_best_state(monkeypatch):
    from vqe_portfolio.baselines import solvers

    score = solvers._objective
    visited = []

    def record(x, *args):
        visited.append(x.copy())
        return score(x, *args)

    monkeypatch.setattr(solvers, "_objective", record)
    result = tabu_search_binary_baseline(
        np.zeros(4),
        np.diag([0, 1, 2, 3]),
        1,
        1,
        restarts=1,
        max_iterations=3,
        tabu_tenure=2,
    )
    # After 0 -> 1, the cheaper reverse 1 -> 0 is tabu, so visit asset 2.
    assert [int(np.argmax(x)) for x in visited] == [0, 1, 2, 0]
    np.testing.assert_array_equal(result.candidate, visited[0])
    assert result.objective == 0
    exhausted = tabu_search_binary_baseline(
        np.zeros(2),
        np.diag([0, 1]),
        1,
        1,
        restarts=1,
        max_iterations=10,
        tabu_tenure=2,
    )
    assert exhausted.status == "tabu_exhausted"
    assert exhausted.objective == 0


@pytest.mark.parametrize("solve", SEARCH_SOLVERS)
@pytest.mark.parametrize("n,k", [(1, 0), (1, 1), (4, 2)])
def test_search_handles_flat_objective_and_single_asset(solve, n, k):
    result = solve(np.zeros(n), np.zeros((n, n)), 0, k, restarts=1, max_iterations=10)
    assert result.candidate.sum() == k
    assert result.objective == 0


@pytest.mark.parametrize("solve", SEARCH_SOLVERS)
@pytest.mark.parametrize("name", ["restarts", "max_iterations"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_search_rejects_invalid_limits(solve, name, value):
    mu, sigma = problem(1)
    with pytest.raises(ValueError, match=name):
        solve(mu, sigma, 1, 3, **{name: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_temperature": 0},
        {"initial_temperature": -1},
        {"initial_temperature": float("nan")},
        {"initial_temperature": float("inf")},
        {"cooling_rate": 0},
        {"cooling_rate": 1},
        {"cooling_rate": -0.1},
        {"cooling_rate": float("nan")},
        {"cooling_rate": float("inf")},
    ],
)
def test_annealing_rejects_invalid_schedule(kwargs):
    mu, sigma = problem(1)
    with pytest.raises(ValueError):
        simulated_annealing_binary_baseline(mu, sigma, 1, 3, **kwargs)


@pytest.mark.parametrize("tenure", [0, -1, 1.5, True])
def test_tabu_rejects_invalid_tenure(tenure):
    mu, sigma = problem(1)
    with pytest.raises(ValueError, match="tabu_tenure"):
        tabu_search_binary_baseline(mu, sigma, 1, 3, tabu_tenure=tenure)
