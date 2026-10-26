from types import SimpleNamespace

import numpy as np
import pytest

from vqe_portfolio.classical import (
    convex_markowitz_baseline,
    local_search_binary_baseline,
    mixed_integer_binary_baseline,
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
    for solve in [mixed_integer_binary_baseline, local_search_binary_baseline]:
        with pytest.raises(ValueError, match="k must"):
            solve(mu, sigma, 1, k)


def test_nonconvex_qp_rejected():
    pytest.importorskip("cvxpy")
    with pytest.raises(ValueError, match="PSD"):
        convex_markowitz_baseline(np.ones(2), -np.eye(2), 1)
