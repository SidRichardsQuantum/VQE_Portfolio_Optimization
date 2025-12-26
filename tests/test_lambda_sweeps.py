import numpy as np

from vqe_portfolio import (
    binary_lambda_sweep,
    fractional_lambda_sweep,
)
from vqe_portfolio.types import LambdaSweepConfig


def test_fractional_lambda_sweep_nontrivial(toy_problem, fractional_cfg):
    mu, Sigma = toy_problem

    sweep = LambdaSweepConfig(
        lambdas=[0.1, 1.0, 5.0],
        steps_per_lambda=15,
        stepsize=0.3,
        warm_start=False,
    )

    res = fractional_lambda_sweep(mu, Sigma, fractional_cfg, sweep)
    allocs = np.asarray(res.allocs_by_lambda)

    # At least two distinct rows
    assert np.unique(np.round(allocs, 6), axis=0).shape[0] > 1


def test_binary_lambda_sweep_nontrivial(toy_problem, binary_cfg):
    mu, Sigma = toy_problem

    sweep = LambdaSweepConfig(
        lambdas=[0.1, 1.0, 5.0],
        steps_per_lambda=15,
        stepsize=0.3,
    )

    res = binary_lambda_sweep(mu, Sigma, binary_cfg, sweep)
    probs = np.asarray(res.probs_by_lambda)

    assert np.unique(np.round(probs, 6), axis=0).shape[0] > 1
