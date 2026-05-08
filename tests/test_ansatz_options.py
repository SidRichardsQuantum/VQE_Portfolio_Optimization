from __future__ import annotations

import numpy as np
import pytest

from vqe_portfolio import BinaryVQEConfig, FractionalVQEConfig
from vqe_portfolio.binary import run_binary_vqe
from vqe_portfolio.fractional import run_fractional_vqe


@pytest.mark.parametrize("ansatz", ["ry_cz", "ry_rz_cz", "strongly_entangling"])
def test_binary_vqe_ansatz_options(toy_problem, ansatz):
    mu, sigma = toy_problem
    cfg = BinaryVQEConfig(
        ansatz=ansatz,
        depth=1,
        steps=2,
        stepsize=0.1,
        log_every=1,
        lam=4.0,
        alpha=2.0,
        k=2,
        shots_train=None,
        shots_sample=32,
        seed=0,
    )

    res = run_binary_vqe(mu, sigma, cfg)

    assert res.x_topk.shape == mu.shape
    assert int(np.sum(res.x_topk)) == cfg.k
    assert res.energy_trace.values


@pytest.mark.parametrize("ansatz", ["ry", "ry_cz", "ry_rz_cz"])
def test_fractional_vqe_ansatz_options(toy_problem, ansatz):
    mu, sigma = toy_problem
    cfg = FractionalVQEConfig(
        ansatz=ansatz,
        depth=1,
        steps=2,
        stepsize=0.1,
        log_every=1,
        lam=4.0,
        shots=None,
        seed=0,
    )

    res = run_fractional_vqe(mu, sigma, cfg)

    assert res.weights.shape == mu.shape
    assert np.all(res.weights >= -1e-10)
    assert abs(float(np.sum(res.weights)) - 1.0) < 1e-6


def test_binary_vqe_rejects_unknown_ansatz(toy_problem):
    mu, sigma = toy_problem
    cfg = BinaryVQEConfig(ansatz="unknown", steps=1, shots_sample=8)

    with pytest.raises(ValueError, match="Unsupported binary ansatz"):
        run_binary_vqe(mu, sigma, cfg)


def test_fractional_vqe_rejects_unknown_ansatz(toy_problem):
    mu, sigma = toy_problem
    cfg = FractionalVQEConfig(ansatz="unknown", steps=1)

    with pytest.raises(ValueError, match="Unsupported fractional ansatz"):
        run_fractional_vqe(mu, sigma, cfg)
