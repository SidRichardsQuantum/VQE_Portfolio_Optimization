"""Replica-exchange balance, complete-state transport, and portfolio incumbents."""

import builtins
import itertools
import json

import numpy as np
import pytest

from vqe_portfolio.baselines import (
    exhaustive_binary_baseline,
    parallel_tempering_binary_baseline,
    simulated_annealing_binary_baseline,
)
from vqe_portfolio.baselines import solvers
from vqe_portfolio.problem import PortfolioProblem


@pytest.mark.parametrize("seed", [0, 15, 40])
@pytest.mark.parametrize("k", [0, 1, 3, 6])
def test_feasible_reproducible_best_incumbents_and_reference_bound(seed, k):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(6, 6))
    mu, sigma = rng.normal(size=6), factors @ factors.T / 6
    problem = PortfolioProblem(mu, sigma, 0.7, "binary_qubo", k)
    before_mu, before_sigma = mu.copy(), sigma.copy()
    published = []

    def publish(x, details):
        assert details["source"] == "parallel-tempering"
        published.append(x.copy())
        x[:] = -10
        details["solver_diagnostics"]["temperatures"][:] = [-1]

    kwargs = dict(
        seed=seed, restarts=2, n_replicas=4, max_iterations=80, exchange_interval=3
    )
    result = parallel_tempering_binary_baseline(
        problem.mu, problem.Sigma, 0.7, k, candidate_callback=publish, **kwargs
    )
    again = parallel_tempering_binary_baseline(mu, sigma, 0.7, k, **kwargs)
    np.testing.assert_array_equal(result.candidate, again.candidate)
    np.testing.assert_array_equal(mu, before_mu)
    np.testing.assert_array_equal(sigma, before_sigma)
    assert result.diagnostics == again.diagnostics
    scores = [problem.evaluate(x) for x in published]
    assert scores and all(score.feasible for score in scores)
    assert all(a.objective > b.objective for a, b in zip(scores, scores[1:]))
    np.testing.assert_array_equal(result.candidate, published[-1])
    assert result.objective == pytest.approx(
        problem.evaluate(result.candidate).objective
    )
    _, exact = exhaustive_binary_baseline(mu, sigma, 0.7, 0, k)
    assert result.objective >= exact - 1e-10
    assert result.lower_bound is None and result.mip_gap is None
    assert result.status == ("trivial" if k in {0, 6} else "iteration_limit")
    diagnostics = result.diagnostics
    assert diagnostics["local_proposals"] == (0 if k in {0, 6} else 2 * 4 * 80)
    assert diagnostics["objective_evaluations"] == (
        1 if k in {0, 6} else 8 + diagnostics["local_accepts"]
    )
    assert diagnostics["temperatures"][0] > 0
    json.dumps(diagnostics, allow_nan=False)


def test_exchange_ratio_matches_product_ensemble_detailed_balance():
    mu = np.array([0.1, -0.2, 0.4, 0.3])
    sigma = np.array([[2, -1, 0, 0], [-1, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 3]])
    energies = []
    for selected in itertools.combinations(range(4), 2):
        x = np.zeros(4)
        x[list(selected)] = 1
        energies.append(x @ sigma @ x - mu @ x)
    cold, hot = 0.5, 2.0
    for e_cold, e_hot in itertools.product(energies, repeat=2):
        ratio = solvers._exchange_log_ratio(float(e_cold), float(e_hot), cold, hot)
        old_weight = np.exp(-e_cold / cold - e_hot / hot)
        new_weight = np.exp(-e_hot / cold - e_cold / hot)
        assert old_weight * min(1, np.exp(ratio)) == pytest.approx(
            new_weight * min(1, np.exp(-ratio)), rel=1e-12, abs=1e-20
        )
    assert solvers._exchange_log_ratio(3.0, 1.0, cold, hot) > 0
    assert solvers._exchange_log_ratio(1.0, 3.0, cold, hot) < 0


def test_metropolis_acceptance_threshold_and_extreme_temperatures():
    class Draw:
        def __init__(self, value):
            self.value = value

        def exponential(self):
            return self.value

    assert solvers._tempering_accept(-1.0, Draw(2.0))
    assert not solvers._tempering_accept(-1.0, Draw(0.5))
    assert solvers._tempering_accept(0.0, None)
    assert solvers._tempering_accept(float("inf"), None)
    assert not solvers._tempering_accept(-float("inf"), Draw(100))
    assert solvers._exchange_log_ratio(1.0, 1.0, 1e-320, 2e-320) == 0
    assert solvers._exchange_log_ratio(1e-310, 2e-310, 1e-310, 2e-310) == pytest.approx(
        -0.5
    )


def test_exchange_moves_states_energies_gradients_and_swap_indices_together(
    monkeypatch,
):
    # Energies of the three one-asset portfolios are 0, 1, 2. First reject two
    # uphill proposals, then exchange E=1 (cold) with E=0 (hot). The next local
    # costs must use the incoming state's indices AND gradient at each fixed T.
    starts = [np.array([0, 1, 0]), np.array([1, 0, 0])]
    monkeypatch.setattr(solvers, "_search_starts", lambda *args: iter(starts))

    class ScriptedRng:
        choices = iter([0, 1, 0, 1, 0, 0, 0, 0])

        def integers(self, high):
            value = next(self.choices)
            assert 0 <= value < high
            return value

        def exponential(self):
            return 0.0  # Reject uphill moves; favorable exchanges are automatic.

    monkeypatch.setattr(np.random, "default_rng", lambda seed: ScriptedRng())
    accept = solvers._tempering_accept
    log_ratios = []

    def record(log_ratio, rng):
        log_ratios.append(log_ratio)
        return accept(log_ratio, rng)

    monkeypatch.setattr(solvers, "_tempering_accept", record)
    result = parallel_tempering_binary_baseline(
        [1, 1, 2],
        np.diag([1, 2, 4]),
        1,
        1,
        restarts=1,
        max_iterations=2,
        n_replicas=2,
        temperatures=[1, 2],
        exchange_interval=1,
    )
    np.testing.assert_allclose(log_ratios, [-1, -1, 0.5, -1, 0.5, 0])
    assert result.diagnostics["exchange_accepts"] == 2
    assert result.diagnostics["local_accepts_by_temperature"] == [0, 1]
    assert result.diagnostics["objective_evaluations"] == 3
    np.testing.assert_array_equal(result.candidate, [1, 0, 0])


def test_alternating_exchange_edges_and_exact_work_counts(monkeypatch):
    evaluated = []
    objective = solvers._objective

    def record(x, *args):
        evaluated.append(x.copy())
        return objective(x, *args)

    monkeypatch.setattr(solvers, "_objective", record)
    ladder = np.array([1, 2, 4, 8], dtype=float)
    result = parallel_tempering_binary_baseline(
        np.zeros(4),
        np.zeros((4, 4)),
        0,
        2,
        restarts=2,
        n_replicas=4,
        max_iterations=6,
        exchange_interval=2,
        temperatures=ladder,
    )
    stats = result.diagnostics
    assert stats["completed_restarts"] == 2 and stats["iterations"] == 12
    assert stats["exchange_attempts_by_edge"] == [4, 2, 4]
    assert stats["exchange_accepts_by_edge"] == [4, 2, 4]
    assert stats["exchange_acceptance_rates"] == [1, 1, 1]
    assert stats["local_proposals_by_temperature"] == [12] * 4
    assert stats["local_acceptance_rates"] == [1] * 4
    assert stats["objective_evaluations"] == len(evaluated) == 8 + 48
    assert stats["temperature_scale_swap_evaluations"] == 0
    assert all(np.count_nonzero(x) == x.sum() == 2 for x in evaluated)
    np.testing.assert_array_equal(ladder, [1, 2, 4, 8])


def test_default_ladder_uses_starting_swap_costs_and_flat_fallback():
    mu, sigma = np.array([0.2, 0.6, 0.4]), np.diag([1, 2, 3])
    start = np.array([0, 1, 0])

    def energy(x):
        return x @ sigma @ x - mu @ x

    scale = max(abs(energy(x) - energy(start)) for x in np.eye(3))
    result = parallel_tempering_binary_baseline(
        mu, sigma, 1, 1, restarts=1, max_iterations=1
    )
    ladder = result.diagnostics["temperatures"]
    np.testing.assert_allclose(ladder, np.geomspace(0.01 * scale, scale, 8))
    assert result.diagnostics["temperature_scale_swap_evaluations"] == 2
    assert result.diagnostics["exchange_attempts"] == 0
    assert result.diagnostics["exchange_acceptance_rates"] == [None] * 7
    flat = parallel_tempering_binary_baseline(
        np.zeros(3), np.zeros((3, 3)), 0, 1, restarts=1, max_iterations=1
    )
    np.testing.assert_allclose(
        flat.diagnostics["temperatures"], np.geomspace(0.01, 1, 8)
    )


def test_hot_replica_escapes_cold_swap_local_minimum():
    mu = np.array([1, 1, 0.1, 0.1])
    sigma = np.array(
        [[5, -3, 0, 0], [-3, 5, 0, 0], [0, 0, 5.5, -5.4], [0, 0, -5.4, 5.5]]
    )
    cold = simulated_annealing_binary_baseline(
        mu, sigma, 1, 2, restarts=1, initial_temperature=1e-12
    )
    tempered = parallel_tempering_binary_baseline(
        mu,
        sigma,
        1,
        2,
        restarts=1,
        n_replicas=2,
        temperatures=[1e-12, 10],
        exchange_interval=1,
    )
    _, exact = exhaustive_binary_baseline(mu, sigma, 1, 0, 2)
    assert tempered.objective < cold.objective
    assert tempered.objective == pytest.approx(exact)
    assert tempered.diagnostics["exchange_accepts"] > 0


@pytest.mark.parametrize(
    "name", ["restarts", "max_iterations", "n_replicas", "exchange_interval"]
)
@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.bool_(False)])
def test_invalid_integer_controls(name, value):
    with pytest.raises(ValueError, match=name):
        parallel_tempering_binary_baseline(
            np.zeros(3), np.eye(3), 1, 1, **{name: value}
        )


def test_requires_multiple_replicas():
    with pytest.raises(ValueError, match="n_replicas"):
        parallel_tempering_binary_baseline(np.zeros(3), np.eye(3), 1, 1, n_replicas=1)


@pytest.mark.parametrize(
    "ladder",
    [
        [1],
        [1, 2, 3],
        [1, 1],
        [2, 1],
        [0, 1],
        [-1, 1],
        [1, np.inf],
        [np.nan, 1],
        [[1, 2]],
        [1, 2j],
    ],
)
def test_invalid_temperature_ladder(ladder):
    with pytest.raises(ValueError, match="temperatures"):
        parallel_tempering_binary_baseline(
            np.zeros(3), np.eye(3), 1, 1, n_replicas=2, temperatures=ladder
        )


@pytest.mark.parametrize("k", [None, -1, 4, 1.5, True])
def test_invalid_cardinality(k):
    with pytest.raises(ValueError, match="k must"):
        parallel_tempering_binary_baseline(np.zeros(3), np.eye(3), 1, k)


def test_numpy_only_and_single_asset(monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"scipy", "cvxpy", "osqp", "pyscipopt"}:
            raise AssertionError(f"Unexpected optional dependency: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for k in (0, 1):
        result = parallel_tempering_binary_baseline([1], [[2]], 1, k)
        assert result.status == "trivial" and result.candidate[0] == k
        assert result.diagnostics["objective_evaluations"] == 1
    assert (
        parallel_tempering_binary_baseline(
            np.zeros(3), np.eye(3), 1, 1, restarts=1, max_iterations=2
        ).candidate.sum()
        == 1
    )
