from dataclasses import replace
import time

import numpy as np
import pytest

from vqe_portfolio import qaoa
from vqe_portfolio.research.budget import paired_budget_trials, summarise_budget_trials
from vqe_portfolio.types import QAOAConfig


def test_untrained_sampler_reuses_production_initial_parameters(monkeypatch):
    mu = np.array([0.1, 0.2, 0.3])
    sigma = np.eye(3) * 0.1
    cfg = QAOAConfig(
        k=1, p=1, steps=0, mixer="xy", xy_initial_state="dicke", seed=7, shots_sample=1
    )
    initial = qaoa.run_qaoa(mu, sigma, cfg).params
    observed = []

    def sampler(h, n, config):
        def sample(params):
            observed.append(np.asarray(params).copy())
            return np.eye(3, dtype=int)[len(observed) - 1]

        return sample

    monkeypatch.setattr(qaoa, "_sampling_qnode", sampler)
    candidates, phases = [], []
    status = qaoa._sample_initial_qaoa(
        mu,
        sigma,
        cfg,
        time.perf_counter() + 5,
        lambda x, details: candidates.append((x, details)),
        phases.append,
    )
    assert status == "exhausted_feasible_space"
    assert len(observed) == 3
    for params in observed:
        np.testing.assert_array_equal(params, initial)
    assert all(details["step"] == 0 for _, details in candidates)
    assert candidates[-1][1]["unique_feasible_samples"] == 3
    assert phases == ["candidate_processing"]
    with pytest.raises(ValueError, match="steps=0"):
        qaoa._sample_initial_qaoa(
            mu, sigma, replace(cfg, steps=1), time.perf_counter(), None
        )


def test_budget_variants_remain_separate_and_pairs_retain_failures():
    base = dict(
        problem_id="same",
        n_assets=4,
        k=2,
        method="qaoa-xy",
        seed=0,
        budget_seconds=10,
        status="completed",
        elapsed_seconds=2,
        overrun_seconds=0,
        reference_contradiction=False,
    )
    trials = [
        {
            **base,
            "variant": "xy-dicke-ring-trained",
            "feasible": True,
            "target_hit": True,
            "time_to_target_seconds": 2,
            "normalised_gap": 0.0,
        },
        {
            **base,
            "variant": "xy-basis-ring",
            "feasible": True,
            "target_hit": False,
            "time_to_target_seconds": None,
            "normalised_gap": 0.2,
        },
        {
            **base,
            "variant": "xy-dicke-ring-untrained",
            "feasible": False,
            "target_hit": False,
            "time_to_target_seconds": None,
            "normalised_gap": None,
        },
    ]
    summary, reliability = summarise_budget_trials(trials)
    assert len(summary) == len(reliability) == 3
    pairs = paired_budget_trials(trials)
    assert len(pairs) == 2
    assert pairs[0]["normalised_improvement"] == pytest.approx(0.2)
    assert pairs[1]["normalised_improvement"] is None
    assert not pairs[1]["comparator_feasible"]
    with pytest.raises(ValueError, match="Duplicate"):
        paired_budget_trials(trials * 2)
