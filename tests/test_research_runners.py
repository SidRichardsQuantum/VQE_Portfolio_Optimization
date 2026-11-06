import json

import pytest


def test_reliability_runner_freezes_settings_and_resumes_without_retraining(
    tmp_path, monkeypatch
):
    from scripts import run_solver_reliability as runner

    args = [
        "--asset-counts",
        "3",
        "--instance-seeds",
        "11",
        "--seeds",
        "0",
        "--steps",
        "3",
        "--shots",
        "8",
        "--output-dir",
        str(tmp_path),
    ]
    assert runner.main(args) == 0
    rows = json.loads((tmp_path / "trials.json").read_text())
    assert len(rows) == 8
    assert all(r["total_training_steps"] == 3 for r in rows)

    def unexpected(*args):
        pytest.fail("A resumed completed run retrained a solver")

    monkeypatch.setattr(runner, "measure_variant", unexpected)
    assert runner.main(args + ["--resume"]) == 0
    assert json.loads((tmp_path / "trials.json").read_text()) == rows
    with pytest.raises(ValueError, match="Resume refused"):
        runner.main(args + ["--steps", "6", "--resume"])
    with pytest.raises(ValueError, match="empty"):
        runner.main(args)


@pytest.mark.parametrize("method", ["swap", "frank-wolfe"])
def test_budget_runner_saves_incumbents_and_refuses_changed_settings(
    tmp_path, monkeypatch, method
):
    from scripts import run_budget_benchmark as runner

    args = [
        "--asset-counts",
        "3",
        "--instance-seeds",
        "11",
        "--seeds",
        "0",
        "--methods",
        method,
        "--budgets",
        "10",
        "--output-dir",
        str(tmp_path),
    ]
    assert runner.main(args) == 0
    rows = json.loads((tmp_path / "trials.json").read_text())
    assert len(rows) == 1 and rows[0]["feasible"]
    assert rows[0]["incumbents"]

    def unexpected(*args):
        pytest.fail("A resumed completed run repeated a solver")

    monkeypatch.setattr(runner, "evaluate_budget_trial", unexpected)
    assert runner.main(args + ["--resume"]) == 0
    with pytest.raises(ValueError, match="Resume refused"):
        runner.main(args + ["--budgets", "1", "--resume"])
    (tmp_path / "trials.json").write_text(json.dumps(rows * 2))
    with pytest.raises(ValueError, match="Duplicate"):
        runner.main(args + ["--resume"])


def test_fresh_budget_config_pairs_initialization_and_refuses_overlap(tmp_path):
    from scripts import run_budget_benchmark as runner

    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "asset_counts": "3",
                "instance_seeds": "404",
                "seeds": "0",
                "variants": "xy-dicke-ring-trained,xy-dicke-ring-untrained",
                "steps": 3,
                "shots": 8,
                "checkpoint_every": 1,
                "budgets": "10",
            }
        )
    )
    output = tmp_path / "fresh"
    args = ["--config", str(config), "--output-dir", str(output)]
    assert runner.main(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    trials = json.loads((output / "trials.json").read_text())
    assert len(trials) == 2
    trained, untrained = trials
    assert trained["problem_id"] == untrained["problem_id"]
    assert trained["solver_config"]["seed"] == untrained["solver_config"]["seed"]
    assert untrained["solver_config"]["steps"] == 0
    assert untrained["timings"]["optimization_seconds"] == 0
    assert untrained["solver_status"] == "exhausted_feasible_space"
    assert (output / "pairs.csv").exists()
    assert (output / "timings.csv").exists()
    assert manifest["settings"]["config_document"] == json.loads(config.read_text())
    assert runner.main(args + ["--resume"]) == 0

    # Identical generated snapshots must not masquerade as fresh instances.
    with pytest.raises(ValueError, match="overlaps"):
        runner.main(
            [
                "--config",
                str(config),
                "--output-dir",
                str(tmp_path / "overlap"),
                "--exclude-manifests",
                str(output / "manifest.json"),
            ]
        )


def test_budget_config_rejects_unknown_settings(tmp_path):
    from scripts import run_budget_benchmark as runner

    config = tmp_path / "settings.json"
    config.write_text('{"shotz": 64}')
    with pytest.raises(SystemExit):
        runner.main(["--config", str(config)])
