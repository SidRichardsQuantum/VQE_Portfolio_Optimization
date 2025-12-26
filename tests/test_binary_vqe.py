import numpy as np

from vqe_portfolio import run_binary_vqe


def test_binary_topk_constraint(toy_problem, binary_cfg):
    mu, Sigma = toy_problem

    res = run_binary_vqe(mu, Sigma, binary_cfg)

    x_topk = np.asarray(res.x_topk, dtype=int)
    x_prob = np.asarray(res.x_prob, dtype=float)

    assert x_topk.sum() == binary_cfg.k
    assert set(np.unique(x_topk)).issubset({0, 1})

    assert np.all(x_prob >= -1e-8)
    assert np.all(x_prob <= 1.0 + 1e-8)

    assert res.energy_trace.steps
    assert res.energy_trace.values
