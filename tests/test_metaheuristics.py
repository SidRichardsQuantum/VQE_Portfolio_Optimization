"""Feasible operators, search mechanics, and independent small-instance checks."""

import numpy as np
import pytest

from vqe_portfolio.baselines import (
    exhaustive_binary_baseline,
    grasp_binary_baseline,
    iterated_local_search_binary_baseline,
    memetic_genetic_binary_baseline,
    variable_neighborhood_search_binary_baseline,
)
from vqe_portfolio.baselines import solvers
from vqe_portfolio.problem import PortfolioProblem


SOLVERS = [
    memetic_genetic_binary_baseline,
    iterated_local_search_binary_baseline,
    grasp_binary_baseline,
    variable_neighborhood_search_binary_baseline,
]


def options(solve):
    if solve is memetic_genetic_binary_baseline:
        return dict(restarts=3, population_size=8, generations=4, max_swaps=100)
    if solve in {
        iterated_local_search_binary_baseline,
        variable_neighborhood_search_binary_baseline,
    }:
        return dict(restarts=3, max_iterations=10, max_swaps=100)
    return dict(restarts=8, max_swaps=100)


@pytest.mark.parametrize("solve", SOLVERS)
@pytest.mark.parametrize("k", [0, 1, 3, 6])
@pytest.mark.parametrize("seed", [0, 15, 40])
def test_feasibility_reproducibility_scoring_and_copied_incumbents(solve, k, seed):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(6, 6))
    mu, sigma = rng.normal(size=6), factors @ factors.T / 6
    original_mu, original_sigma = mu.copy(), sigma.copy()
    problem = PortfolioProblem(mu, sigma, 0.7, "binary_qubo", k)
    published = []

    def publish(x, details):
        assert details["source"] in {"memetic", "ils", "grasp", "vns"}
        published.append(x.copy())
        x[:] = -100

    result = solve(
        mu, sigma, 0.7, k, seed=seed, candidate_callback=publish, **options(solve)
    )
    repeat = solve(problem.mu, problem.Sigma, 0.7, k, seed=seed, **options(solve))
    np.testing.assert_array_equal(result.candidate, repeat.candidate)
    np.testing.assert_array_equal(mu, original_mu)
    np.testing.assert_array_equal(sigma, original_sigma)
    assert result.objective == repeat.objective
    evaluations = [problem.evaluate(x) for x in published]
    assert evaluations and all(e.feasible for e in evaluations)
    assert all(a.objective > b.objective for a, b in zip(evaluations, evaluations[1:]))
    np.testing.assert_array_equal(result.candidate, published[-1])
    assert result.objective == pytest.approx(
        problem.evaluate(result.candidate).objective
    )
    _, exact = exhaustive_binary_baseline(mu, sigma, 0.7, 0, k)
    assert result.objective >= exact - 1e-10
    if solve is not grasp_binary_baseline:
        start = np.zeros(6, dtype=int)
        start[np.argsort(-mu)[:k]] = 1
        assert result.objective <= problem.evaluate(start).objective + 1e-10
    for i in np.flatnonzero(result.candidate):
        for j in np.flatnonzero(1 - result.candidate):
            neighbour = result.candidate.copy()
            neighbour[i], neighbour[j] = 0, 1
            assert problem.evaluate(neighbour).objective >= result.objective - 1e-10
    assert result.lower_bound is None and result.mip_gap is None
    expected = {
        memetic_genetic_binary_baseline: "generation_limit",
        iterated_local_search_binary_baseline: "iteration_limit",
        grasp_binary_baseline: "restart_limit",
        variable_neighborhood_search_binary_baseline: "iteration_limit",
    }[solve]
    assert result.status == ("trivial" if k in {0, 6} else expected)


@pytest.fixture
def barrier():
    return (
        np.array([1.0, 1.0, 0.1, 0.1]),
        np.array([[5, -3, 0, 0], [-3, 5, 0, 0], [0, 0, 5.5, -5.4], [0, 0, -5.4, 5.5]]),
    )


@pytest.mark.parametrize("solve", SOLVERS)
def test_reaches_better_basin_than_the_top_return_swap_start(solve, barrier):
    mu, sigma = barrier
    settings = options(solve)
    if solve is grasp_binary_baseline:
        settings["rcl_alpha"] = 1
    result = solve(mu, sigma, 1, 2, seed=0, **settings)
    _, exact = exhaustive_binary_baseline(mu, sigma, 1, 0, 2)
    assert result.objective == pytest.approx(exact)
    assert result.objective < 2  # Objective of the first pair, a swap-local optimum.


@pytest.mark.parametrize("k", [0, 1, 3, 6])
def test_crossover_and_mutation_preserve_cardinality_and_parents(k):
    rng = np.random.default_rng(7)
    for _ in range(20):
        parents = np.zeros((2, 6), dtype=int)
        for x in parents:
            x[rng.choice(6, k, replace=False)] = 1
        original = parents.copy()
        child = solvers._subset_crossover(*parents, rng)
        assert child.sum() == k
        assert np.all(child[(parents[0] & parents[1]) == 1] == 1)
        assert np.all(child[(parents[0] | parents[1]) == 0] == 0)
        mutated = solvers._perturb_selection(child, rng, 2)
        assert mutated.sum() == k
        assert np.count_nonzero(mutated != child) == 2 * min(2, k, 6 - k)
        np.testing.assert_array_equal(parents, original)


def test_memetic_evolution_uses_crossover_mutation_and_local_refinement(monkeypatch):
    crossovers, mutations, refinements = [], [], []
    crossover = solvers._subset_crossover
    perturb = solvers._perturb_selection
    refine = solvers._BinaryLocalSearch.refine

    def cross(first, second, rng):
        child = crossover(first, second, rng)
        crossovers.append(child.copy())
        return child

    def mutate(x, rng, swaps):
        child = perturb(x, rng, swaps)
        mutations.append(np.count_nonzero(child != x))
        return child

    def local(self, x, source, **details):
        refinements.append(details)
        return refine(self, x, source, **details)

    monkeypatch.setattr(solvers, "_subset_crossover", cross)
    monkeypatch.setattr(solvers, "_perturb_selection", mutate)
    monkeypatch.setattr(solvers._BinaryLocalSearch, "refine", local)
    memetic_genetic_binary_baseline(
        np.arange(6.0),
        np.eye(6),
        0.7,
        3,
        restarts=1,
        population_size=4,
        generations=3,
        mutation_rate=1,
    )
    assert crossovers and all(x.sum() == 3 for x in crossovers)
    assert mutations and all(distance == 2 for distance in mutations)
    assert len(refinements) == 4 + 3 * 3
    assert any(d["generation"] == 3 for d in refinements)


def test_ils_rejects_worse_refined_perturbations(monkeypatch, barrier):
    first, second = np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1])
    proposed, current = iter([second, first, first]), []

    def perturb(x, rng, swaps):
        current.append(x.copy())
        return next(proposed).copy()

    monkeypatch.setattr(solvers, "_perturb_selection", perturb)
    result = iterated_local_search_binary_baseline(
        *barrier, 1, 2, restarts=1, max_iterations=3
    )
    np.testing.assert_array_equal(current, [first, second, second])
    np.testing.assert_array_equal(result.candidate, second)


def test_vns_advances_wraps_and_resets_neighborhood_after_improvement(monkeypatch):
    first, second = np.array([1, 1, 1, 0, 0, 0]), np.array([0, 0, 0, 1, 1, 1])
    mu = np.array([1, 1, 1, 0.1, 0.1, 0.1])
    sigma = np.zeros((6, 6))
    sigma[:3, :3] = -1.5
    np.fill_diagonal(sigma[:3, :3], 5)
    sigma[3:, 3:] = -2.7
    np.fill_diagonal(sigma[3:, 3:], 5.5)
    proposed = iter([first, second, first, first, first, first])
    neighborhoods, current = [], []

    def shake(x, rng, swaps):
        neighborhoods.append(swaps)
        current.append(x.copy())
        return next(proposed).copy()

    monkeypatch.setattr(solvers, "_perturb_selection", shake)
    result = variable_neighborhood_search_binary_baseline(
        mu,
        sigma,
        1,
        3,
        restarts=1,
        max_iterations=6,
        max_neighborhood=3,
    )
    assert neighborhoods == [1, 2, 1, 2, 3, 1]
    np.testing.assert_array_equal(
        current, [first, first, second, second, second, second]
    )
    np.testing.assert_array_equal(result.candidate, second)


def test_vns_clamps_neighborhood_to_the_available_assets(monkeypatch):
    perturb = solvers._perturb_selection
    sizes = []

    def shake(x, rng, swaps):
        sizes.append(swaps)
        return perturb(x, rng, swaps)

    monkeypatch.setattr(solvers, "_perturb_selection", shake)
    result = variable_neighborhood_search_binary_baseline(
        np.zeros(5),
        np.eye(5),
        1,
        1,
        restarts=1,
        max_iterations=5,
        max_neighborhood=99,
    )
    assert sizes == [1] * 5
    assert result.candidate.sum() == 1
    assert "resolved_max_neighborhood=1" in result.message


@pytest.mark.parametrize("name", ["max_iterations", "max_neighborhood"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_vns_rejects_invalid_settings(name, value):
    with pytest.raises(ValueError, match=name):
        variable_neighborhood_search_binary_baseline(
            np.zeros(6), np.eye(6), 1, 3, **{name: value}
        )


def test_grasp_construction_uses_covariance_marginal_costs():
    mu = np.array([10.0, 9.0, 8.0])
    sigma = np.array([[1.0, 8.0, 0], [8.0, 100.0, 0], [0, 0, 2.0]])
    published = []
    grasp_binary_baseline(
        mu,
        sigma,
        1,
        2,
        restarts=1,
        rcl_alpha=0,
        candidate_callback=lambda x, details: published.append(x),
    )
    # Marginal costs choose asset 0, then asset 2 rather than top-return asset 1.
    np.testing.assert_array_equal(published[0], [1, 0, 1])


def test_grasp_alpha_one_random_construction_is_independent_of_costs():
    initial = []
    for mu in [np.arange(6.0), -np.arange(6.0)]:
        published = []
        grasp_binary_baseline(
            mu,
            np.eye(6),
            1,
            3,
            seed=4,
            restarts=1,
            rcl_alpha=1,
            candidate_callback=lambda x, details: published.append(x),
        )
        initial.append(published[0])
    np.testing.assert_array_equal(*initial)


@pytest.mark.parametrize("solve", SOLVERS)
@pytest.mark.parametrize("n,k", [(1, 0), (1, 1), (3, 1)])
def test_flat_objective_and_small_feasible_spaces_terminate(solve, n, k):
    result = solve(np.zeros(n), np.zeros((n, n)), 0, k, **options(solve))
    assert result.candidate.sum() == k and result.objective == 0


@pytest.mark.parametrize("solve", SOLVERS)
@pytest.mark.parametrize("k", [-1, 7, 1.5, True])
def test_invalid_cardinality(solve, k):
    with pytest.raises(ValueError, match="k must"):
        solve(np.zeros(6), np.eye(6), 1, k)


@pytest.mark.parametrize("solve", SOLVERS)
@pytest.mark.parametrize("name", ["restarts", "max_swaps"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_common_limits(solve, name, value):
    with pytest.raises(ValueError, match=name):
        solve(np.zeros(6), np.eye(6), 1, 3, **{name: value})


@pytest.mark.parametrize(
    "solve,kwargs",
    [
        (memetic_genetic_binary_baseline, {"population_size": 1}),
        (memetic_genetic_binary_baseline, {"population_size": 2.5}),
        (memetic_genetic_binary_baseline, {"generations": 0}),
        (memetic_genetic_binary_baseline, {"tournament_size": 0}),
        (memetic_genetic_binary_baseline, {"mutation_rate": -1}),
        (memetic_genetic_binary_baseline, {"mutation_rate": 1.1}),
        (memetic_genetic_binary_baseline, {"mutation_rate": np.nan}),
        (memetic_genetic_binary_baseline, {"mutation_rate": np.inf}),
        (iterated_local_search_binary_baseline, {"max_iterations": 0}),
        (iterated_local_search_binary_baseline, {"perturbation_size": 0}),
        (iterated_local_search_binary_baseline, {"perturbation_size": True}),
        (grasp_binary_baseline, {"rcl_alpha": -1}),
        (grasp_binary_baseline, {"rcl_alpha": 1.1}),
        (grasp_binary_baseline, {"rcl_alpha": np.nan}),
        (grasp_binary_baseline, {"rcl_alpha": np.inf}),
    ],
)
def test_invalid_algorithm_settings(solve, kwargs):
    with pytest.raises(ValueError):
        solve(np.zeros(6), np.eye(6), 1, 3, **kwargs)
