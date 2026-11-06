"""Simplex projection, PGD convergence, and independent fractional references."""

import numpy as np
import pytest

from vqe_portfolio.baselines import (
    exact_long_only_markowitz_baseline,
    projected_gradient_descent_baseline,
)
from vqe_portfolio.baselines.solvers import _project_simplex
from vqe_portfolio.problem import PortfolioProblem


@pytest.mark.parametrize(
    "vector,expected",
    [
        ([-1, 0, 2], [0, 0, 1]),
        ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
        ([-2, -2, -2], [1 / 3, 1 / 3, 1 / 3]),
        ([2, 1], [1, 0]),
        ([1e15, 1e15, 1e15], [1 / 3, 1 / 3, 1 / 3]),
        ([20], [1]),
    ],
)
def test_simplex_projection_known_solutions(vector, expected):
    projected = _project_simplex(np.array(vector, dtype=float))
    np.testing.assert_allclose(projected, expected, atol=1e-12)


def test_simplex_projection_satisfies_kkt_and_translation_invariance():
    rng = np.random.default_rng(12)
    for _ in range(20):
        vector = rng.normal(size=12)
        projected = _project_simplex(vector)
        assert projected.min() >= 0
        assert projected.sum() == pytest.approx(1)
        active = projected > 0
        threshold = (vector - projected)[active][0]
        np.testing.assert_allclose((vector - projected)[active], threshold, atol=1e-12)
        assert np.all(vector[~active] <= threshold + 1e-12)
        np.testing.assert_allclose(
            _project_simplex(vector + 1000), projected, atol=1e-12
        )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lam", [0, 0.7, 5])
def test_pgd_matches_small_active_set_reference_and_retains_feasible_copies(seed, lam):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(5, 5))
    mu, sigma = rng.normal(size=5), factors @ factors.T / 5 + 0.1 * np.eye(5)
    frozen = PortfolioProblem(mu, sigma, lam, "fractional_simplex")
    published = []

    def publish(w, details):
        assert details["source"] == "pgd"
        published.append(w.copy())
        w[:] = -10

    result = projected_gradient_descent_baseline(
        frozen.mu, frozen.Sigma, lam, candidate_callback=publish
    )
    again = projected_gradient_descent_baseline(mu, sigma, lam)
    np.testing.assert_array_equal(result.candidate, again.candidate)
    scores = [frozen.evaluate(w) for w in published]
    assert scores and all(s.feasible for s in scores)
    assert all(a.objective > b.objective for a, b in zip(scores, scores[1:]))
    np.testing.assert_array_equal(result.candidate, published[-1])
    assert result.objective == pytest.approx(
        frozen.evaluate(result.candidate).objective
    )
    _, expected = exact_long_only_markowitz_baseline(mu, sigma, lam)
    assert result.objective == pytest.approx(expected, abs=1e-7)
    assert result.objective <= frozen.evaluate(np.full(5, 0.2)).objective
    assert result.lower_bound is None and result.mip_gap is None
    assert result.status in {"converged", "numerical_limit"}
    gradient = 2 * lam * sigma @ result.candidate - mu
    gap = (gradient - gradient.min()) @ result.candidate
    assert gap >= 0
    if result.status == "converged":
        assert gap <= 1e-8
    else:
        assert gap > 1e-8


def test_pgd_backtracks_an_oversized_initial_step():
    mu, sigma = np.array([0.1, 0.4, 0.2]), np.diag([1.0, 2.0, 3.0])
    result = projected_gradient_descent_baseline(
        mu, sigma, 1, stepsize=100, tolerance=1e-7
    )
    _, expected = exact_long_only_markowitz_baseline(mu, sigma, 1)
    assert result.status == "converged"
    assert result.objective == pytest.approx(expected, abs=1e-7)


def test_pgd_tiny_step_does_not_falsely_report_convergence():
    result = projected_gradient_descent_baseline(
        np.array([0.0, 0.1, 1.0]), np.eye(3), 1, stepsize=1e-100, max_iterations=2
    )
    assert result.status == "iteration_limit"
    np.testing.assert_allclose(result.candidate, np.full(3, 1 / 3))


def test_pgd_iteration_and_line_search_limits_keep_a_feasible_incumbent():
    mu, sigma = np.array([0.0, 0.1, 1.0]), np.eye(3)
    problem = PortfolioProblem(mu, sigma, 1, "fractional_simplex")
    limited = projected_gradient_descent_baseline(
        mu, sigma, 1, max_iterations=1, stepsize=0.001
    )
    assert limited.status == "iteration_limit"
    assert problem.evaluate(limited.candidate).feasible
    failed_search = projected_gradient_descent_baseline(mu, sigma, 1, stepsize=1e100)
    assert failed_search.status == "line_search_limit"
    assert problem.evaluate(failed_search.candidate).feasible


@pytest.mark.parametrize(
    "mu,sigma,lam",
    [
        (np.zeros(4), np.zeros((4, 4)), 0),
        (np.zeros(4), np.eye(4), 1),
        (np.array([1.0, 3.0, 2.0]), np.zeros((3, 3)), 4),
        (np.array([2.0]), np.array([[0.1]]), 3),
    ],
)
def test_pgd_flat_singular_and_single_asset_problems(mu, sigma, lam):
    result = projected_gradient_descent_baseline(mu, sigma, lam)
    assert result.status == "converged"
    assert (
        PortfolioProblem(mu, sigma, lam, "fractional_simplex")
        .evaluate(result.candidate)
        .feasible
    )
    _, expected = exact_long_only_markowitz_baseline(mu, sigma, lam)
    assert result.objective == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_pgd_rejects_invalid_iteration_limit(value):
    with pytest.raises(ValueError, match="max_iterations"):
        projected_gradient_descent_baseline(
            np.zeros(3), np.eye(3), 1, max_iterations=value
        )


@pytest.mark.parametrize("name", ["tolerance", "stepsize"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_pgd_rejects_invalid_numerical_settings(name, value):
    with pytest.raises(ValueError, match=name):
        projected_gradient_descent_baseline(np.zeros(3), np.eye(3), 1, **{name: value})


def test_pgd_rejects_non_psd_covariance():
    with pytest.raises(ValueError, match="PSD"):
        projected_gradient_descent_baseline(np.zeros(3), -np.eye(3), 1)
