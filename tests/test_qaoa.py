from __future__ import annotations

import numpy as np
import pytest

from vqe_portfolio.qaoa import qaoa_lambda_sweep, run_qaoa
from vqe_portfolio.types import LambdaSweepConfig, QAOAConfig


def _toy_problem():
    mu = np.array([0.10, 0.20, 0.15], dtype=float)
    sigma = np.array(
        [
            [0.05, 0.01, 0.00],
            [0.01, 0.06, 0.01],
            [0.00, 0.01, 0.04],
        ],
        dtype=float,
    )
    return mu, sigma


def test_run_qaoa_smoke_x_mixer():
    mu, sigma = _toy_problem()

    cfg = QAOAConfig(
        p=1,
        steps=3,
        stepsize=0.1,
        k=2,
        lam=1.0,
        alpha=5.0,
        mixer="x",
        seed=0,
        shots_train=None,
        shots_sample=128,
        log_every=1,
    )

    res = run_qaoa(mu=mu, Sigma=sigma, cfg=cfg)

    n = len(mu)

    assert res.params.shape == (cfg.p, 2)
    assert res.gammas.shape == (cfg.p,)
    assert res.betas.shape == (cfg.p,)

    assert len(res.cost_trace.steps) == len(res.cost_trace.values)
    assert len(res.cost_trace.steps) >= 1

    assert res.state_probs.shape == (2**n,)
    assert np.isclose(np.sum(res.state_probs), 1.0)

    assert res.x_prob.shape == (n,)
    assert res.x_round.shape == (n,)
    assert res.x_topk.shape == (n,)
    assert res.x_mode.shape == (n,)

    assert np.all(res.x_prob >= 0.0)
    assert np.all(res.x_prob <= 1.0)

    assert set(np.asarray(np.unique(res.x_round)).tolist()).issubset({0, 1})
    assert set(np.asarray(np.unique(res.x_topk)).tolist()).issubset({0, 1})
    assert set(np.asarray(np.unique(res.x_mode)).tolist()).issubset({0, 1})

    assert int(np.sum(res.x_topk)) == cfg.k

    assert isinstance(res.sample_counts, dict)
    assert len(res.sample_counts) >= 1
    assert sum(res.sample_counts.values()) == cfg.shots_sample

    if res.x_best_feasible is not None:
        assert res.x_best_feasible.shape == (n,)
        assert set(np.asarray(np.unique(res.x_best_feasible)).tolist()).issubset({0, 1})
        assert int(np.sum(res.x_best_feasible)) == cfg.k

    assert res.lambdas is None
    assert res.probs_by_lambda is None


def test_run_qaoa_smoke_xy_mixer():
    mu, sigma = _toy_problem()

    cfg = QAOAConfig(
        p=1,
        steps=3,
        stepsize=0.1,
        k=2,
        lam=1.0,
        alpha=5.0,
        mixer="xy",
        seed=0,
        shots_train=None,
        shots_sample=128,
        log_every=1,
    )

    res = run_qaoa(mu=mu, Sigma=sigma, cfg=cfg)

    n = len(mu)

    assert res.params.shape == (cfg.p, 2)
    assert res.state_probs.shape == (2**n,)
    assert np.isclose(np.sum(res.state_probs), 1.0)

    assert res.x_prob.shape == (n,)
    assert res.x_topk.shape == (n,)
    assert int(np.sum(res.x_topk)) == cfg.k

    assert isinstance(res.sample_counts, dict)
    assert sum(res.sample_counts.values()) == cfg.shots_sample


def test_qaoa_lambda_sweep_shapes():
    mu, sigma = _toy_problem()

    cfg = QAOAConfig(
        p=1,
        steps=2,
        stepsize=0.1,
        k=2,
        lam=1.0,
        alpha=5.0,
        mixer="x",
        seed=0,
        shots_train=None,
        shots_sample=64,
        log_every=1,
    )

    sweep = LambdaSweepConfig(
        lambdas=[0.5, 1.0, 2.0],
        steps_per_lambda=2,
        stepsize=0.1,
        warm_start=False,
    )

    res = qaoa_lambda_sweep(mu=mu, Sigma=sigma, cfg=cfg, sweep=sweep)

    n = len(mu)

    assert res.lambdas is not None
    assert res.probs_by_lambda is not None

    assert res.lambdas.shape == (3,)
    assert res.probs_by_lambda.shape == (3, n)

    assert np.all(res.probs_by_lambda >= 0.0)
    assert np.all(res.probs_by_lambda <= 1.0)


def test_run_qaoa_rejects_bad_k():
    mu, sigma = _toy_problem()

    cfg = QAOAConfig(
        p=1,
        steps=2,
        stepsize=0.1,
        k=0,
        lam=1.0,
        alpha=5.0,
        mixer="x",
    )

    with pytest.raises(ValueError, match="cfg.k"):
        run_qaoa(mu=mu, Sigma=sigma, cfg=cfg)


def test_run_qaoa_rejects_bad_p():
    mu, sigma = _toy_problem()

    cfg = QAOAConfig(
        p=0,
        steps=2,
        stepsize=0.1,
        k=2,
        lam=1.0,
        alpha=5.0,
        mixer="x",
    )

    with pytest.raises(ValueError, match="cfg.p"):
        run_qaoa(mu=mu, Sigma=sigma, cfg=cfg)


def test_run_qaoa_rejects_bad_steps():
    mu, sigma = _toy_problem()

    cfg = QAOAConfig(
        p=1,
        steps=0,
        stepsize=0.1,
        k=2,
        lam=1.0,
        alpha=5.0,
        mixer="x",
    )

    with pytest.raises(ValueError, match="cfg.steps"):
        run_qaoa(mu=mu, Sigma=sigma, cfg=cfg)


def test_run_qaoa_rejects_bad_stepsize():
    mu, sigma = _toy_problem()

    cfg = QAOAConfig(
        p=1,
        steps=2,
        stepsize=0.0,
        k=2,
        lam=1.0,
        alpha=5.0,
        mixer="x",
    )

    with pytest.raises(ValueError, match="cfg.stepsize"):
        run_qaoa(mu=mu, Sigma=sigma, cfg=cfg)


def test_run_qaoa_rejects_bad_mixer():
    mu, sigma = _toy_problem()

    cfg = QAOAConfig(
        p=1,
        steps=2,
        stepsize=0.1,
        k=2,
        lam=1.0,
        alpha=5.0,
        mixer="bad",
    )

    with pytest.raises(ValueError, match="cfg.mixer"):
        run_qaoa(mu=mu, Sigma=sigma, cfg=cfg)


def test_run_qaoa_rejects_nonsquare_sigma():
    mu = np.array([0.1, 0.2, 0.3], dtype=float)
    sigma = np.array([[1.0, 0.1]], dtype=float)

    cfg = QAOAConfig(
        p=1,
        steps=2,
        stepsize=0.1,
        k=2,
        lam=1.0,
        alpha=5.0,
        mixer="x",
    )

    with pytest.raises(ValueError, match="incompatible with mu shape"):
        run_qaoa(mu=mu, Sigma=sigma, cfg=cfg)


def test_run_qaoa_rejects_mu_sigma_dimension_mismatch():
    mu = np.array([0.1, 0.2], dtype=float)
    sigma = np.eye(3, dtype=float)

    cfg = QAOAConfig(
        p=1,
        steps=2,
        stepsize=0.1,
        k=1,
        lam=1.0,
        alpha=5.0,
        mixer="x",
    )

    with pytest.raises(ValueError, match="incompatible with mu shape"):
        run_qaoa(mu=mu, Sigma=sigma, cfg=cfg)


def test_qaoa_lambda_sweep_rejects_empty_lambdas():
    mu, sigma = _toy_problem()

    cfg = QAOAConfig(
        p=1,
        steps=2,
        stepsize=0.1,
        k=2,
        lam=1.0,
        alpha=5.0,
        mixer="x",
    )

    sweep = LambdaSweepConfig(
        lambdas=[],
        steps_per_lambda=2,
        stepsize=0.1,
        warm_start=False,
    )

    with pytest.raises(ValueError, match="non-empty 1D sequence"):
        qaoa_lambda_sweep(mu=mu, Sigma=sigma, cfg=cfg, sweep=sweep)
