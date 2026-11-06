"""The binary relaxation keeps sum(z)=K and rounds before independent scoring."""

import numpy as np
import pytest

from vqe_portfolio.baselines import (
    exhaustive_binary_baseline,
    relaxation_rounding_binary_baseline,
)
from vqe_portfolio.problem import PortfolioProblem

cp = pytest.importorskip("cvxpy")
pytest.importorskip("osqp")


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("k", [0, 1, 3, 6])
@pytest.mark.parametrize("lam", [0, 0.7])
def test_real_relaxation_returns_reproducible_feasible_locally_refined_portfolios(
    seed, k, lam
):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(6, 6))
    mu, sigma = rng.normal(size=6), factors @ factors.T / 6
    frozen = PortfolioProblem(mu, sigma, lam, "binary_qubo", k)
    before = (mu.copy(), sigma.copy())
    published = []

    def publish(x, details):
        assert details["source"] == "qp-round"
        published.append(x.copy())
        x[:] = -99

    result = relaxation_rounding_binary_baseline(
        mu, sigma, lam, k, candidate_callback=publish
    )
    again = relaxation_rounding_binary_baseline(frozen.mu, frozen.Sigma, lam, k)
    np.testing.assert_array_equal(result.candidate, again.candidate)
    np.testing.assert_array_equal(mu, before[0])
    np.testing.assert_array_equal(sigma, before[1])
    evaluations = [frozen.evaluate(x) for x in published]
    assert evaluations and all(e.feasible for e in evaluations)
    assert all(a.objective > b.objective for a, b in zip(evaluations, evaluations[1:]))
    np.testing.assert_array_equal(result.candidate, published[-1])
    assert result.objective == pytest.approx(
        frozen.evaluate(result.candidate).objective
    )
    assert result.objective <= evaluations[0].objective + 1e-10
    _, exact = exhaustive_binary_baseline(mu, sigma, lam, 0, k)
    assert result.objective >= exact - 1e-10
    for i in np.flatnonzero(result.candidate):
        for j in np.flatnonzero(1 - result.candidate):
            neighbour = result.candidate.copy()
            neighbour[i], neighbour[j] = 0, 1
            assert frozen.evaluate(neighbour).objective >= result.objective - 1e-10
    assert result.lower_bound is None and result.mip_gap is None
    assert result.status == ("trivial" if k in {0, 6} else "rounded_and_refined")


def test_relaxation_retains_quadratic_diagonal_terms_and_sum_k(monkeypatch):
    solve = cp.Problem.solve
    observed = []

    def record(self, **kwargs):
        result = solve(self, **kwargs)
        observed.append(self.variables()[0].value.copy())
        return result

    monkeypatch.setattr(cp.Problem, "solve", record)
    mu, sigma = np.zeros(4), np.diag([1.0, 2.0, 3.0, 4.0])
    result = relaxation_rounding_binary_baseline(mu, sigma, 1, 2)
    # Interior optimum is c / diagonal, with c chosen so the entries sum to K.
    expected = 2 / np.sum(1 / np.diag(sigma)) / np.diag(sigma)
    np.testing.assert_allclose(observed[0], expected, atol=1e-8)
    assert observed[0].sum() == pytest.approx(2)
    assert np.all((observed[0] > 0) & (observed[0] < 1))
    assert result.objective == pytest.approx(3)
    assert "relaxation_status=optimal" in result.message
    assert "relaxation_objective=" in result.message
    assert result.lower_bound is None  # A numerical primal value is not a certificate.


def stub_relaxation(monkeypatch, raw, status="optimal"):
    def solve(self, **kwargs):
        self._status = status
        self.variables()[0]._value = None if raw is None else np.asarray(raw)

    monkeypatch.setattr(cp.Problem, "solve", solve)


def test_rounding_ties_use_asset_index_order(monkeypatch):
    stub_relaxation(monkeypatch, [0.5, 0.5, 0.5, 0.5])
    result = relaxation_rounding_binary_baseline(np.zeros(4), np.eye(4), 1, 2)
    np.testing.assert_array_equal(result.candidate, [1, 1, 0, 0])
    assert result.status == "rounded_and_refined"


@pytest.mark.parametrize("native_status", ["optimal_inaccurate", "user_limit"])
def test_swap_refinement_improves_the_rounded_portfolio(monkeypatch, native_status):
    stub_relaxation(monkeypatch, [0.8, 0.7, 0.3, 0.2], native_status)
    published = []
    result = relaxation_rounding_binary_baseline(
        np.zeros(4),
        np.diag([10.0, 1.0, 2.0, 3.0]),
        1,
        2,
        candidate_callback=lambda x, details: published.append(x),
    )
    np.testing.assert_array_equal(published[0], [1, 1, 0, 0])
    np.testing.assert_array_equal(result.candidate, [0, 1, 1, 0])
    assert result.objective == pytest.approx(3)
    assert f"relaxation_status={native_status}" in result.message
    assert result.status == "rounded_and_refined"
    assert result.lower_bound is None and result.mip_gap is None


@pytest.mark.parametrize(
    "raw",
    [
        [np.nan, 0.5, 0.5, 1],
        [np.inf, 0.5, 0.5, 1],
        [-0.1, 0.7, 0.7, 0.7],
        [1.1, 0.3, 0.3, 0.3],
        [0.2, 0.2, 0.2, 0.2],
        [[0.5, 0.5, 0.5, 0.5]],
        [0.5, 0.5, 1],
        [complex(0.5, 0.1), 0.5, 0.5, 0.5],
    ],
)
def test_invalid_relaxation_is_not_silently_rounded(monkeypatch, raw):
    stub_relaxation(monkeypatch, raw, "user_limit")
    published = []
    result = relaxation_rounding_binary_baseline(
        np.zeros(4),
        np.eye(4),
        1,
        2,
        candidate_callback=lambda *args: published.append(args),
    )
    assert result.status == "invalid_relaxation"
    assert result.candidate is None and result.objective is None
    assert "relaxation_status=user_limit" in result.message
    assert not published


def test_missing_relaxation_output_retains_native_status(monkeypatch):
    stub_relaxation(monkeypatch, None, "user_limit")
    result = relaxation_rounding_binary_baseline(np.zeros(4), np.eye(4), 1, 2)
    assert result.status == "user_limit"
    assert result.candidate is None and result.objective is None


def test_small_numerical_bound_violations_are_clipped_before_rounding(monkeypatch):
    stub_relaxation(monkeypatch, [-1e-10, 1 + 1e-10, 0.5, 0.5])
    result = relaxation_rounding_binary_baseline(np.zeros(4), np.eye(4), 1, 2)
    np.testing.assert_array_equal(result.candidate, [0, 1, 1, 0])


def test_nonconvex_covariance_is_rejected():
    with pytest.raises(ValueError, match="PSD"):
        relaxation_rounding_binary_baseline(np.ones(3), -np.eye(3), 1, 1)


def test_singular_covariance_and_single_asset_are_supported():
    result = relaxation_rounding_binary_baseline(
        np.array([1.0, 3.0, 2.0]), np.zeros((3, 3)), 4, 2
    )
    assert result.objective == pytest.approx(-5)
    for k in [0, 1]:
        result = relaxation_rounding_binary_baseline(np.array([1.0]), np.eye(1), 1, k)
        assert result.status == "trivial"
        np.testing.assert_array_equal(result.candidate, [k])


def test_solver_exceptions_propagate(monkeypatch):
    def failed(*args, **kwargs):
        raise cp.error.SolverError("OSQP failed")

    monkeypatch.setattr(cp.Problem, "solve", failed)
    with pytest.raises(cp.error.SolverError, match="OSQP failed"):
        relaxation_rounding_binary_baseline(np.zeros(4), np.eye(4), 1, 2)
