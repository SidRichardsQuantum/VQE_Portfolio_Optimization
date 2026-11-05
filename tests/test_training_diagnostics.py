import itertools
import json

import numpy as np
import pytest

from vqe_portfolio.binary import run_binary_vqe
from vqe_portfolio.problem import PortfolioProblem
from vqe_portfolio.qaoa import run_qaoa
from vqe_portfolio.research import training_diagnostics as diagnostic
from vqe_portfolio.research.validation import ValidationConfig, prepare_instances
from vqe_portfolio.types import BinaryVQEConfig, QAOAConfig


def small_config():
    return ValidationConfig(
        asset_counts=(3,),
        instance_seeds=(11,),
        solver_seeds=(0,),
        steps=3,
        shots=16,
        methods=diagnostic.DIAGNOSTIC_METHODS,
    )


def test_basis_order_feasibility_and_target_use_original_objective():
    problem = PortfolioProblem([0.1, 0.3, 0.2], np.eye(3) * 0.01, 1, "binary_qubo", 1)
    bits, feasible, target, scores = diagnostic.target_basis(problem, -0.29, 1e-3)
    np.testing.assert_array_equal(bits, list(itertools.product((0, 1), repeat=3)))
    np.testing.assert_array_equal(np.flatnonzero(feasible), [1, 2, 4])
    np.testing.assert_array_equal(np.flatnonzero(target), [2])  # bitstring 010
    for x, score in zip(bits, scores):
        assert score == pytest.approx(x @ problem.Sigma @ x - problem.mu @ x)
    # A large tolerance includes all feasible states, never infeasible ones.
    _, feasible, target, _ = diagnostic.target_basis(problem, -0.29, 100)
    np.testing.assert_array_equal(target, feasible)
    with pytest.raises(ValueError, match="reference disagrees"):
        diagnostic.target_basis(problem, -0.1, 1e-3)


def test_probability_metrics_include_infeasible_mass_and_zero_denominator():
    feasible = np.array([False, True, True, False])
    target = np.array([False, True, False, False])
    metrics = diagnostic.distribution_metrics(
        [0.1, 0.2, 0.3, 0.4], feasible, target, np.array([4, 2, 1, 5])
    )
    assert metrics["p_feasible"] == pytest.approx(0.5)
    assert metrics["p_target"] == pytest.approx(0.2)
    assert metrics["p_target_given_feasible"] == pytest.approx(0.4)
    assert metrics["expected_penalized_energy"] == pytest.approx(3.1)
    zero = diagnostic.distribution_metrics([1, 0, 0, 0], feasible, target, np.ones(4))
    assert zero["p_target_given_feasible"] is None
    for invalid in ([0, 0, 0, 0], [float("nan"), 0, 0, 1], [-0.1, 0.2, 0.3, 0.6]):
        with pytest.raises(ValueError, match="Invalid exact"):
            diagnostic.distribution_metrics(invalid, feasible, target, np.ones(4))


@pytest.mark.parametrize("method", diagnostic.DIAGNOSTIC_METHODS)
def test_training_matches_public_solver_and_preserves_initial_parameters(method):
    config = small_config()
    instance = prepare_instances(config)[0]
    problem = PortfolioProblem.from_dict(instance["snapshot"])
    result = diagnostic.train_circuit(problem, method, 0, config)
    common = {
        "steps": 3,
        "log_every": 3,
        "lam": problem.lam,
        "alpha": config.alpha,
        "k": problem.k,
        "seed": 0,
        "shots_sample": 16,
    }
    if method == "binary-vqe":
        cfg = BinaryVQEConfig(**common, depth=config.binary_depth, stepsize=0.3)
        public = run_binary_vqe(problem.mu, problem.Sigma, cfg)
        zero_cfg = BinaryVQEConfig(
            **{**common, "steps": 0}, depth=config.binary_depth, stepsize=0.3
        )
        initial = run_binary_vqe(problem.mu, problem.Sigma, zero_cfg)
        np.testing.assert_allclose(result["initial_params"], initial.params, atol=1e-12)
        # Independent measurement of marginals catches basis-order mistakes.
        bits = np.array(list(itertools.product((0, 1), repeat=3)))
        np.testing.assert_allclose(
            np.array(result["after_probabilities"]) @ bits, public.x_prob, atol=1e-12
        )
    else:
        cfg = QAOAConfig(
            **common, p=config.qaoa_p, mixer=method.removeprefix("qaoa-"), stepsize=0.2
        )
        public = run_qaoa(problem.mu, problem.Sigma, cfg)
        np.testing.assert_allclose(
            result["after_probabilities"], public.state_probs, atol=1e-12
        )
    np.testing.assert_allclose(result["trained_params"], public.params, atol=1e-12)
    assert not np.allclose(result["initial_params"], result["trained_params"])
    assert sum(result["before_probabilities"]) == pytest.approx(1)
    assert sum(result["after_probabilities"]) == pytest.approx(1)
    # Re-running from the same seed reproduces both ends of the pair.
    repeated = diagnostic.train_circuit(problem, method, 0, config)
    assert repeated["initial_params"] == result["initial_params"]
    np.testing.assert_allclose(
        repeated["after_probabilities"], result["after_probabilities"], atol=1e-12
    )


def test_pair_matches_direct_mass_sums_and_errors_are_preserved(monkeypatch):
    config = small_config()
    instance = prepare_instances(config)[0]
    row = diagnostic.measure_pair("pair", instance, "qaoa-xy", 0, config)
    assert row["error"] == ""
    indices = row["target_basis_indices"]
    before = np.array(row["before_probabilities"])[indices].sum()
    after = np.array(row["after_probabilities"])[indices].sum()
    assert row["before_p_target"] == pytest.approx(before)
    assert row["delta_p_target"] == pytest.approx(after - before)
    assert row["before_p_feasible"] == pytest.approx(1)
    assert row["after_p_feasible"] == pytest.approx(1)

    def failed(*args):
        raise RuntimeError("training failed")

    monkeypatch.setattr(diagnostic, "train_circuit", failed)
    failed_row = diagnostic.measure_pair("failed", instance, "qaoa-xy", 0, config)
    assert failed_row["error"] == "RuntimeError: training failed"
    summary = diagnostic.summarise_pairs([row, failed_row])[0]
    assert summary["pair_count"] == 2 and summary["successful_pairs"] == 1
    assert summary["error_count"] == 1


def test_summary_averages_paired_changes_and_weights_instances_equally():
    rows = []
    for problem_id, before, after in (
        ("a", 0.1, 0.2),
        ("a", 0.1, 0.2),
        ("b", 0.5, 0.1),
    ):
        rows.append(
            {
                "n_assets": 4,
                "k": 2,
                "method": "test",
                "problem_id": problem_id,
                "error": "",
                "before_p_target": before,
                "after_p_target": after,
                "delta_p_target": after - before,
                "before_p_feasible": 1,
                "after_p_feasible": 1,
                "before_p_target_given_feasible": before,
                "after_p_target_given_feasible": after,
                "uniform_feasible_p_target": 1 / 6,
            }
        )
    summary = diagnostic.summarise_pairs(rows)[0]
    assert summary["mean_delta_p_target"] == pytest.approx(-0.15)
    assert summary["mean_before_p_target"] == pytest.approx(0.3)
    assert summary["improved_pairs"] == 2 and summary["worsened_pairs"] == 1


def test_runner_checkpoint_resume_and_pilot_change_refusal(tmp_path, monkeypatch):
    from scripts import run_training_diagnostic as runner

    config = small_config()
    pilot = tmp_path / "pilot.json"
    pilot.write_text(
        json.dumps({"config": config.to_dict(), "instances": prepare_instances(config)})
    )
    output = tmp_path / "out"
    args = ["--pilot-manifest", str(pilot), "--output-dir", str(output)]
    assert runner.main(args) == 0
    rows = json.loads((output / "pairs.json").read_text())
    assert len(rows) == 3
    assert (output / "training_probability.png").exists()
    (output / "pairs.json").write_text(json.dumps(rows[:2]))
    original = runner.measure_pair
    calls = []

    def record(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(runner, "measure_pair", record)
    assert runner.main(args + ["--resume"]) == 0
    assert calls == [rows[2]["trial_id"]]
    assert json.loads((output / "pairs.json").read_text())[:2] == rows[:2]
    with pytest.raises(ValueError, match="nonempty"):
        runner.main(args)
    pilot.write_text(pilot.read_text() + "\n")
    with pytest.raises(ValueError, match="Resume refused"):
        runner.main(args + ["--resume"])
