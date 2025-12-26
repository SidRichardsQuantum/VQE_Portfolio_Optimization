import numpy as np

from vqe_portfolio import run_fractional_vqe


def test_fractional_simplex_constraints(toy_problem, fractional_cfg):
    mu, Sigma = toy_problem

    res = run_fractional_vqe(mu, Sigma, fractional_cfg)
    w = np.asarray(res.weights, dtype=float)

    assert w.ndim == 1
    assert len(w) == len(mu)

    # Simplex constraints
    assert np.all(w >= -1e-10)
    assert abs(w.sum() - 1.0) < 1e-6

    # Trace recorded
    assert len(res.cost_trace.steps) > 0
    assert len(res.cost_trace.values) == len(res.cost_trace.steps)
