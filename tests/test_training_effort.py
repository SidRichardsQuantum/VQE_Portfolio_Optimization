import json
from dataclasses import replace

import numpy as np
import pytest

from vqe_portfolio import training_diagnostics as diagnostic
from vqe_portfolio import training_effort as effort
from vqe_portfolio.problem import PortfolioProblem
from vqe_portfolio.validation import ValidationConfig, prepare_instances, trial_plan


def small_config():
    return ValidationConfig(
        asset_counts=(3,),
        instance_seeds=(11,),
        solver_seeds=(0,),
        steps=1,
        shots=16,
        methods=diagnostic.DIAGNOSTIC_METHODS,
    )


@pytest.mark.parametrize("method", diagnostic.DIAGNOSTIC_METHODS)
def test_checkpoint_trajectory_matches_independent_runs_and_uses_one_adam(
    monkeypatch, method
):
    config = replace(small_config(), steps=4)
    instance = prepare_instances(config)[0]
    problem = PortfolioProblem.from_dict(instance["snapshot"])
    original = diagnostic.adam_optimize
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs["steps"])
        return original(*args, **kwargs)

    monkeypatch.setattr(diagnostic, "adam_optimize", record)
    result = diagnostic.train_checkpoints(problem, method, 0, config, (0, 2, 4))
    assert calls == [4]  # no optimiser restart at step 2
    states = result["checkpoints"]
    assert [s["step"] for s in states] == [0, 2, 4]
    times = [s["training_runtime_seconds"] for s in states]
    assert times[0] == 0 and 0 < times[1] <= times[2]
    for step in (2, 4):
        independent = diagnostic.train_circuit(
            problem, method, 0, replace(config, steps=step)
        )
        state = next(s for s in states if s["step"] == step)
        np.testing.assert_allclose(
            state["params"], independent["trained_params"], atol=1e-12
        )
        np.testing.assert_allclose(
            state["probabilities"], independent["after_probabilities"], atol=1e-12
        )
        np.testing.assert_allclose(
            states[0]["params"], independent["initial_params"], atol=1e-12
        )


@pytest.mark.parametrize(
    "steps",
    [(0,), (1, 4), (0, 2, 2, 4), (0, -1, 4), (0, 2.5, 4), (0, True, 4), (0, 2, 5)],
)
def test_invalid_checkpoints_are_rejected(steps):
    with pytest.raises(ValueError):
        diagnostic.validate_checkpoints(steps, 4)


def test_trajectory_metrics_source_verification_and_failure_preservation(monkeypatch):
    source_config = small_config()
    instance = prepare_instances(source_config)[0]
    source = diagnostic.measure_pair("test", instance, "binary-vqe", 0, source_config)
    config = replace(source_config, steps=3)
    row = effort.measure_trajectory(
        "test", instance, "binary-vqe", 0, config, (0, 1, 3), (16, 64, 512), source
    )
    assert row["error"] == ""
    assert row["source_checkpoints_verified"] == [0, 1]
    for state in row["checkpoints"]:
        for m in (16, 64, 512):
            assert state["success_probabilities"][str(m)] == pytest.approx(
                1 - (1 - state["p_target"]) ** m
            )
    source["trained_params"] = (np.array(source["trained_params"]) + 0.5).tolist()
    mismatch = effort.measure_trajectory(
        "bad", instance, "binary-vqe", 0, config, (0, 1, 3), (16,), source
    )
    assert "disagrees" in mismatch["error"]
    assert "checkpoints" not in mismatch

    def failed(*args):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(effort, "train_checkpoints", failed)
    failed_row = effort.measure_trajectory(
        "failed", instance, "binary-vqe", 0, config, (0, 1, 3), (16,)
    )
    rows, summaries, paired = effort.summarise_effort(
        [row, failed_row], (0, 1, 3), (16,)
    )
    assert len(rows) == 6 and len(summaries) == 3 and len(paired) == 3
    assert all(
        s["trajectory_count"] == 2
        and s["successful_trajectories"] == 1
        and s["error_count"] == 1
        for s in summaries
    )
    assert all(r["p_target"] is None for r in rows if r["error"])
    direct = row["checkpoints"][2]["p_target"] - row["checkpoints"][1]["p_target"]
    assert paired[-1]["mean_delta_p_target"] == pytest.approx(direct)


def test_instance_weighted_means_and_nonlinear_success_are_preserved():
    trajectories = []
    for problem_id, ps in (
        ("a", (0.0, 0.2, 0.4)),
        ("a", (1.0, 0.8, 0.6)),
        ("b", (0.5, 0.6, 0.7)),
    ):
        states = [
            {
                "step": step,
                "p_target": p,
                "p_feasible": 1,
                "p_target_given_feasible": p,
                "expected_penalized_energy": 0,
                "training_runtime_seconds": float(step),
                "success_probabilities": {"16": 1 - (1 - p) ** 16},
            }
            for step, p in zip((0, 25, 100), ps)
        ]
        trajectories.append(
            {
                "trial_id": str(len(trajectories)),
                "problem_id": problem_id,
                "instance_seed": 0,
                "n_assets": 4,
                "k": 2,
                "method": "binary-vqe",
                "seed": 0,
                "error": "",
                "checkpoints": states,
            }
        )
    _, summaries, paired = effort.summarise_effort(trajectories, (0, 25, 100), (16,))
    assert summaries[0]["mean_success_16"] == pytest.approx((0.5 + (1 - 0.5**16)) / 2)
    assert summaries[0]["mean_success_16"] != pytest.approx(1 - 0.5**16)
    assert paired[-1]["mean_delta_p_target"] == pytest.approx(0.05)
    assert paired[-1]["improved_pairs"] == 2 and paired[-1]["worsened_pairs"] == 1


def test_runner_resumes_completed_trajectories_and_refuses_changed_settings(
    tmp_path, monkeypatch
):
    from scripts import run_training_effort as runner

    config = small_config()
    instances = prepare_instances(config)
    records = [
        diagnostic.measure_pair(key, instance, method, seed, config)
        for key, instance, method, seed in trial_plan(instances, config)
    ]
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study": "initial_vs_trained_binary_distributions",
                "config": config.to_dict(),
                "instances": instances,
            }
        )
    )
    (source / "pairs.json").write_text(json.dumps(records))
    output = tmp_path / "study"
    args = [
        "--diagnostic-dir",
        str(source),
        "--output-dir",
        str(output),
        "--steps",
        "0,1,2",
    ]
    assert runner.main(args) == 0
    trajectories = json.loads((output / "trajectories.json").read_text())
    assert len(trajectories) == 3
    assert all(r["source_checkpoints_verified"] == [0, 1] for r in trajectories)
    assert (output / "training_pairs.png").exists()
    (output / "trajectories.json").write_text(json.dumps(trajectories[:2]))
    original = runner.measure_trajectory
    calls = []

    def record(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(runner, "measure_trajectory", record)
    assert runner.main(args + ["--resume"]) == 0
    assert calls == [trajectories[-1]["trial_id"]]
    assert (
        json.loads((output / "trajectories.json").read_text())[:2] == trajectories[:2]
    )
    with pytest.raises(ValueError, match="nonempty"):
        runner.main(args)
    with pytest.raises(ValueError, match="Resume refused"):
        runner.main(args + ["--steps", "0,1,3", "--resume"])
