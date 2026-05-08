from __future__ import annotations

import numpy as np

from vqe_portfolio.comparison import (
    binary_objective,
    equal_weight_baseline,
    exact_long_only_markowitz_baseline,
    exhaustive_binary_baseline,
    feasible_sample_rate,
    fractional_objective,
    minimum_variance_binary_baseline,
    selected_equal_weight,
    top_return_baseline,
)


def test_binary_baselines_return_feasible_selections(toy_problem):
    mu, sigma = toy_problem
    lam = 4.0
    alpha = 2.0
    k = 2

    for baseline in [
        exhaustive_binary_baseline,
        top_return_baseline,
        minimum_variance_binary_baseline,
    ]:
        x, obj = baseline(mu, sigma, lam=lam, alpha=alpha, k=k)
        assert x.shape == mu.shape
        assert int(x.sum()) == k
        assert obj == binary_objective(x, mu, sigma, lam=lam, alpha=alpha, k=k)


def test_fractional_baselines_return_simplex_weights(toy_problem):
    mu, sigma = toy_problem
    lam = 4.0

    for baseline in [
        equal_weight_baseline,
        exact_long_only_markowitz_baseline,
    ]:
        w, obj = baseline(mu, sigma, lam=lam)
        assert w.shape == mu.shape
        assert np.all(w >= -1e-10)
        assert abs(w.sum() - 1.0) < 1e-8
        assert obj == fractional_objective(w, mu, sigma, lam=lam)


def test_feasible_sample_rate():
    counts = {"110": 3, "101": 1, "111": 2, "000": 4}
    assert feasible_sample_rate(counts, k=2) == 0.4


def test_selected_equal_weight():
    np.testing.assert_allclose(
        selected_equal_weight(np.array([1, 0, 1])), [0.5, 0.0, 0.5]
    )
