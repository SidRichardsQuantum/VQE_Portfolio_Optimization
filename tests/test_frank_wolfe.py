"""Away-step geometry, exact line search, and independent portfolio references."""

import builtins

import numpy as np
import pytest

from vqe_portfolio.baselines import (
    convex_markowitz_baseline,
    exact_long_only_markowitz_baseline,
    frank_wolfe_baseline,
    projected_gradient_descent_baseline,
)
from vqe_portfolio.problem import PortfolioProblem


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lam", [0, 0.7, 5])
def test_matches_exact_reference_and_publishes_independent_incumbents(seed, lam):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(5, 5))
    mu, sigma = rng.normal(size=5), factors @ factors.T / 5 + 0.1 * np.eye(5)
    before_mu, before_sigma = mu.copy(), sigma.copy()
    problem = PortfolioProblem(mu, sigma, lam, "fractional_simplex")
    published = []

    def publish(w, details):
        assert details["source"] == "frank-wolfe"
        published.append(w.copy())
        w[:] = -10

    result = frank_wolfe_baseline(
        problem.mu, problem.Sigma, lam, candidate_callback=publish
    )
    again = frank_wolfe_baseline(mu, sigma, lam)
    np.testing.assert_array_equal(result.candidate, again.candidate)
    np.testing.assert_array_equal(mu, before_mu)
    np.testing.assert_array_equal(sigma, before_sigma)
    evaluations = [problem.evaluate(w) for w in published]
    assert evaluations and all(e.feasible for e in evaluations)
    assert all(a.objective > b.objective for a, b in zip(evaluations, evaluations[1:]))
    np.testing.assert_array_equal(result.candidate, published[-1])
    assert result.objective == pytest.approx(
        problem.evaluate(result.candidate).objective
    )
    _, expected = exact_long_only_markowitz_baseline(mu, sigma, lam)
    assert result.objective == pytest.approx(expected, abs=1e-7)
    assert result.lower_bound is None and result.mip_gap is None
    gradient = 2 * lam * sigma @ result.candidate - mu
    gap = (gradient - gradient.min()) @ result.candidate
    assert result.status in {"converged", "numerical_limit"}
    if result.status == "converged":
        assert gap <= 1e-8
    else:
        assert gap > 1e-8
    assert "first_order_gap=" in result.message


def test_forward_away_and_drop_steps_obey_algorithm_and_exact_line_search():
    # This PSD instance first adds vertices, then takes an interior away step,
    # then drops an active vertex at the feasible bound.
    rng = np.random.default_rng(0)
    factors = rng.normal(size=(4, 4))
    mu, sigma = rng.normal(size=4), factors @ factors.T
    records = []
    result = frank_wolfe_baseline(
        mu,
        sigma,
        1,
        max_iterations=6,
        candidate_callback=lambda w, details: records.append((w, details)),
    )
    assert result.status == "iteration_limit"
    assert len(records) == 7
    assert records[0][1]["step_type"] == "initial"
    np.testing.assert_array_equal(records[0][0], [0, 0, 0, 1])
    step_types = set()
    drops = 0
    for (previous, _), (current, details) in zip(records, records[1:]):
        gradient = 2 * sigma @ previous - mu
        forward = int(np.argmin(gradient))
        active = np.flatnonzero(previous > 0)
        away = int(active[np.argmax(gradient[active])])
        forward_gap = gradient @ previous - gradient[forward]
        away_gap = gradient[away] - gradient @ previous
        atom = details["atom"]
        step_types.add(details["step_type"])
        if details["step_type"] == "away":
            assert away_gap > forward_gap
            assert atom == away
            direction = previous - np.eye(4)[atom]
            bound = previous[atom] / (1 - previous[atom])
        else:
            assert forward_gap >= away_gap
            assert atom == forward
            direction = np.eye(4)[atom] - previous
            bound = 1
        expected_step = min(
            bound, -(gradient @ direction) / (2 * direction @ sigma @ direction)
        )
        step = details["step_size"]
        assert details["max_step_size"] == pytest.approx(bound)
        assert 0 < step <= bound
        assert step == pytest.approx(expected_step)
        np.testing.assert_allclose(current, previous + step * direction, atol=1e-14)
        if details["step_type"] == "away" and step == bound:
            drops += 1
            assert previous[atom] > 0 and current[atom] == 0
    assert step_types == {"forward", "away"}
    assert drops == 1
    assert "forward_steps=3; away_steps=3; drop_steps=1" in result.message
    np.testing.assert_array_equal(result.candidate, records[-1][0])


@pytest.mark.parametrize(
    "mu,sigma,lam,expected",
    [
        (np.zeros(4), np.zeros((4, 4)), 0, [1, 0, 0, 0]),
        (np.zeros(4), np.eye(4), 1, [0.25] * 4),
        ([1, 3, 2], np.zeros((3, 3)), 4, [0, 1, 0]),
        ([2], [[0.1]], 3, [1]),
        ([0.5, 0], [[4, 1], [1, 0.5]], 1, [0, 1]),
        ([0, 0.8], np.eye(2), 1, [0.3, 0.7]),
        (np.zeros(3), np.diag([1, 0, 0]), 1, [0, 1, 0]),
    ],
)
def test_flat_singular_boundary_and_known_interior_solutions(mu, sigma, lam, expected):
    result = frank_wolfe_baseline(mu, sigma, lam)
    assert result.status == "converged"
    np.testing.assert_allclose(result.candidate, expected, atol=1e-8)
    assert (
        PortfolioProblem(mu, sigma, lam, "fractional_simplex")
        .evaluate(result.candidate)
        .feasible
    )


def test_iteration_limit_retains_feasible_incumbent_without_false_convergence():
    problem = PortfolioProblem(np.zeros(4), np.eye(4), 1, "fractional_simplex")
    result = frank_wolfe_baseline(problem.mu, problem.Sigma, 1, max_iterations=1)
    assert result.status == "iteration_limit"
    assert problem.evaluate(result.candidate).feasible
    assert "iterations=1;" in result.message
    gradient = 2 * result.candidate
    assert (gradient - gradient.min()) @ result.candidate > 1e-8


def test_numpy_only_execution(monkeypatch):
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"cvxpy", "scipy", "osqp", "pyscipopt"}:
            raise AssertionError(f"Unexpected optional import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    result = frank_wolfe_baseline(np.zeros(3), np.eye(3), 1)
    np.testing.assert_allclose(result.candidate, np.full(3, 1 / 3))


@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.bool_(False)])
def test_rejects_invalid_iteration_limit(value):
    with pytest.raises(ValueError, match="max_iterations"):
        frank_wolfe_baseline(np.zeros(3), np.eye(3), 1, max_iterations=value)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_rejects_invalid_tolerance(value):
    with pytest.raises(ValueError, match="tolerance"):
        frank_wolfe_baseline(np.zeros(3), np.eye(3), 1, tolerance=value)


def test_rejects_non_psd_covariance():
    with pytest.raises(ValueError, match="PSD"):
        frank_wolfe_baseline(np.zeros(3), -np.eye(3), 1)


@pytest.mark.parametrize("singular", [False, True])
def test_agrees_with_osqp_and_pgd(singular):
    pytest.importorskip("cvxpy")
    pytest.importorskip("osqp")
    rng = np.random.default_rng(13)
    factors = rng.normal(size=(5, 2 if singular else 5))
    sigma, mu = factors @ factors.T, rng.normal(size=5)
    fw = frank_wolfe_baseline(mu, sigma, 0.7)
    pgd = projected_gradient_descent_baseline(mu, sigma, 0.7)
    osqp = convex_markowitz_baseline(mu, sigma, 0.7)
    assert fw.objective == pytest.approx(osqp.objective, abs=1e-7)
    assert fw.objective == pytest.approx(pgd.objective, abs=1e-7)
