import numpy as np

from vqe_portfolio.frontier import (
    binary_frontier_from_probs,
    fractional_frontier_from_allocs,
)


def test_binary_frontier_shapes():
    mu = np.array([0.1, 0.2, 0.15])
    Sigma = np.eye(3)
    lambdas = np.array([1.0, 2.0, 3.0])
    probs = np.array([
        [0.9, 0.1, 0.2],
        [0.2, 0.8, 0.1],
        [0.3, 0.2, 0.7],
    ])

    frontier = binary_frontier_from_probs(mu, Sigma, lambdas, probs, k=1)

    assert frontier.weights.shape == (3, 3)
    assert frontier.risks.shape == (3,)
    assert frontier.returns.shape == (3,)
    assert frontier.lambdas.shape == (3,)


def test_fractional_frontier_shapes():
    mu = np.array([0.1, 0.2, 0.15])
    Sigma = np.eye(3)
    lambdas = np.array([1.0, 2.0])
    allocs = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.5, 0.3],
    ])

    frontier = fractional_frontier_from_allocs(mu, Sigma, lambdas, allocs)

    assert frontier.weights.shape == (2, 3)
    assert frontier.risks.shape == (2,)
    assert frontier.returns.shape == (2,)
