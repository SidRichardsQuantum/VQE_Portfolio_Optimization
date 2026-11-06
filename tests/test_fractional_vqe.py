import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
import pytest

from vqe_portfolio import (
    run_variational_allocator,
    VariationalAllocatorConfig,
)
from vqe_portfolio.fractional import angles_to_weights, ALLOCATOR_SMOOTHING
from vqe_portfolio.problem import PortfolioProblem


@pytest.mark.parametrize(
    "z",
    [
        [1.0, 1.0, 1.0],
        [1.0 - 1e-14, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, 0.0, 1.0],
        [1.0],
        [1.0 + 1e-12, -1.0 - 1e-12],
    ],
)
def test_allocator_normalisation_is_feasible_at_zero_and_near_zero_mass(z):
    z = qnp.array(z, requires_grad=True)
    weights = angles_to_weights(z)
    problem = PortfolioProblem(
        np.zeros(len(z)), np.eye(len(z)), 1, "fractional_simplex"
    )
    assert problem.evaluate(weights).feasible
    assert np.all(np.asarray(weights) >= 0)
    assert float(qnp.sum(weights)) == pytest.approx(1, abs=1e-14)
    gradient = qml.grad(
        lambda values: qnp.dot(angles_to_weights(values), qnp.arange(len(z)))
    )(z)
    assert np.all(np.isfinite(gradient))
    if np.all(np.asarray(z) == 1):
        np.testing.assert_allclose(weights, np.full(len(z), 1 / len(z)), atol=1e-15)


def test_allocator_gradient_matches_analytic_derivative_and_finite_differences():
    z = qnp.array([0.2, -0.4, 0.7], requires_grad=True)
    coefficients = qnp.array([0.3, -0.2, 0.8], requires_grad=False)

    def cost(values):
        return qnp.dot(coefficients, angles_to_weights(values))

    gradient = np.asarray(qml.grad(cost)(z))
    mass = (1 - np.asarray(z)) / 2
    denominator = mass.sum() + ALLOCATOR_SMOOTHING
    numerator = np.dot(coefficients, mass + ALLOCATOR_SMOOTHING / len(z))
    expected = (-0.5 * coefficients * denominator + 0.5 * numerator) / denominator**2
    np.testing.assert_allclose(gradient, expected, atol=1e-12)
    finite_difference = []
    for i in range(len(z)):
        direction = np.eye(len(z))[i] * 1e-6
        finite_difference.append(
            (float(cost(z + direction)) - float(cost(z - direction))) / 2e-6
        )
    np.testing.assert_allclose(gradient, finite_difference, atol=1e-8)
    assert np.linalg.norm(gradient) > 0
    np.testing.assert_allclose(
        qml.grad(lambda values: qnp.sum(angles_to_weights(values)))(z), 0, atol=1e-12
    )


@pytest.mark.parametrize(
    "z", [[], [[1.0, 1.0]], [np.nan], [np.inf], [complex(0, 1)], [1.1], [-1.1]]
)
def test_allocator_rejects_invalid_expectations(z):
    with pytest.raises(ValueError):
        angles_to_weights(z)


@pytest.mark.parametrize("shots", [None, 16])
def test_allocator_trains_on_the_all_zero_bit_state_without_invalid_weights(
    monkeypatch, shots
):
    import vqe_portfolio.fractional as fractional

    monkeypatch.setattr(
        fractional.np.random,
        "uniform",
        lambda *args, **kwargs: np.zeros(kwargs["size"]),
    )
    mu, sigma = np.array([0.1, 0.2, 0.3]), np.eye(3)
    cfg = VariationalAllocatorConfig(steps=2, log_every=1, shots=shots, seed=7)
    result = run_variational_allocator(mu, sigma, cfg)
    np.testing.assert_allclose(result.weights, np.full(3, 1 / 3))
    assert np.all(np.isfinite(result.cost_trace.values))
    assert (
        PortfolioProblem(mu, sigma, cfg.lam, "fractional_simplex")
        .evaluate(result.weights)
        .feasible
    )
