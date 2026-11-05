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


def test_budget_runner_saves_incumbents_and_refuses_changed_settings(
    tmp_path, monkeypatch
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
        "swap",
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
