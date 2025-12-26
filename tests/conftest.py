import numpy as np
import pytest

from vqe_portfolio.types import BinaryVQEConfig, FractionalVQEConfig


@pytest.fixture
def toy_problem():
    """
    Small, well-conditioned toy portfolio.
    """
    mu = np.array([0.12, 0.10, 0.07])
    Sigma = np.array([
        [0.10, 0.02, 0.01],
        [0.02, 0.08, 0.01],
        [0.01, 0.01, 0.05],
    ])
    return mu, Sigma


@pytest.fixture
def fractional_cfg():
    return FractionalVQEConfig(
        steps=25,
        stepsize=0.3,
        log_every=5,
        lam=4.0,
        shots=None,
        seed=0,
    )


@pytest.fixture
def binary_cfg():
    return BinaryVQEConfig(
        depth=2,
        steps=25,
        stepsize=0.3,
        log_every=5,
        lam=4.0,
        alpha=2.0,
        k=2,
        shots_train=None,
        shots_sample=200,
        seed=0,
    )
