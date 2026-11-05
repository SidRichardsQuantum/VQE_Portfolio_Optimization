import copy
import csv
import itertools
import json

import numpy as np
import pytest

from vqe_portfolio.sampling_allowance import (
    sampling_rows,
    success_probability,
    summarise_sampling,
    validate_draw_counts,
)
from vqe_portfolio.validation import ValidationConfig, prepare_instances, trial_plan


@pytest.fixture
def evidence():
    config = ValidationConfig(
        asset_counts=(3,),
        instance_seeds=(11, 22),
        solver_seeds=(0, 1, 2),
        methods=("binary-vqe",),
    )
    instances = prepare_instances(config)
    records = []
    for key, instance, method, seed in trial_plan(instances, config):
        snapshot = instance["snapshot"]
        mu, sigma = np.array(snapshot["mu"]), np.array(snapshot["Sigma"])
        scale = abs(config.lam) * snapshot["k"] ** 2 * np.max(abs(sigma)) + snapshot[
            "k"
        ] * np.max(abs(mu))
        targets = []
        for index, x in enumerate(itertools.product((0, 1), repeat=3)):
            x = np.array(x)
            if (
                x.sum() == snapshot["k"]
                and (
                    config.lam * x @ sigma @ x
                    - mu @ x
                    - instance["reference_objective"]
                )
                / scale
                <= config.target_gap
            ):
                targets.append(index)
        record = {
            "trial_id": key,
            "problem_id": instance["problem_id"],
            "instance_seed": instance["instance_seed"],
            "method": method,
            "seed": seed,
            "n_assets": 3,
            "k": snapshot["k"],
            "error": "",
            "target_basis_indices": targets,
        }
        for phase, p in (("before", seed / 2), ("after", 0.5)):
            probabilities = np.zeros(8)
            probabilities[0] = 1 - p  # 000 is infeasible and must still consume draws
            probabilities[targets[0]] = p
            record[f"{phase}_probabilities"] = probabilities.tolist()
            record[f"{phase}_p_target"] = p
        records.append(record)
    manifest = {
        "schema_version": 1,
        "study": "initial_vs_trained_binary_distributions",
        "config": config.to_dict(),
        "instances": instances,
    }
    return manifest, records


def test_probability_formula_endpoints_tiny_values_and_repeated_failure():
    assert success_probability(0, 512) == 0
    assert success_probability(1, 512) == 1
    assert success_probability(0.2, 1) == pytest.approx(0.2)
    assert success_probability(0.2, 3) == pytest.approx(1 - 0.8**3)
    assert success_probability(1e-69, 512) == pytest.approx(512e-69, rel=1e-14, abs=0)
    # Independent enumeration of the only failure outcome among 2**4 sequences.
    outcomes = list(itertools.product((0, 1), repeat=4))
    expected = sum(any(x) for x in outcomes) / len(outcomes)
    assert success_probability(0.5, 4) == pytest.approx(expected)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probabilities_are_rejected(probability):
    with pytest.raises(ValueError):
        success_probability(probability, 16)


@pytest.mark.parametrize("draws", [(), (0,), (-1,), (1.5,), (True,), (16, 16)])
def test_invalid_allowances_are_rejected(draws):
    with pytest.raises(ValueError):
        validate_draw_counts(draws)


def test_transform_each_run_before_averaging_and_deduplicate_baseline(evidence):
    manifest, records = evidence
    pairs, baselines = sampling_rows(manifest, records, (16, 64, 512))
    assert len(pairs) == 18 and len(baselines) == 6
    summaries, classical = summarise_sampling(pairs, baselines)
    assert len(summaries) == 3 and len(classical) == 3
    initial = (
        success_probability(0, 16)
        + success_probability(0.5, 16)
        + success_probability(1, 16)
    ) / 3
    trained = success_probability(0.5, 16)
    summary = summaries[0]
    assert summary["mean_initial_success_probability"] == pytest.approx(initial)
    assert summary["mean_trained_success_probability"] == pytest.approx(trained)
    assert summary["mean_training_change"] == pytest.approx(trained - initial)
    assert summary["mean_initial_success_probability"] != pytest.approx(
        success_probability(0.5, 16)
    )
    for baseline in baselines:
        assert (
            baseline["p_target"]
            == baseline["target_count"] / baseline["feasible_count"]
        )
    # Feasibility conditioning would incorrectly turn a half-target/half-000
    # distribution into a certain hit. All raw draws count instead.
    half = next(r for r in pairs if r["seed"] == 1 and r["draws"] == 16)
    assert half["initial_success_probability"] < 1


def test_failure_coverage_is_preserved_with_blank_probabilities(evidence):
    manifest, records = evidence
    records[0]["error"] = "RuntimeError: source training failed"
    pairs, baselines = sampling_rows(manifest, records, (16,))
    assert pairs[0]["initial_success_probability"] is None
    assert pairs[0]["trained_success_probability"] is None
    summaries, classical = summarise_sampling(pairs, baselines)
    summary = summaries[0]
    assert summary["pair_count"] == 6 and summary["successful_pairs"] == 5
    assert summary["quality_instance_count"] == 2 and summary["error_count"] == 1
    # Average the two instance means, not all five surviving runs together.
    trained = success_probability(0.5, 16)
    expected = ((trained + 1) / 2 + (0 + trained + 1) / 3) / 2
    assert summary["mean_initial_success_probability"] == pytest.approx(expected)
    assert classical[0]["instance_count"] == 2


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "duplicate",
        "metadata",
        "probabilities",
        "target",
        "stored_probability",
        "reference",
    ],
)
def test_corrupted_or_incomplete_evidence_is_rejected(evidence, corruption):
    manifest, records = copy.deepcopy(evidence)
    if corruption == "missing":
        records.pop()
    elif corruption == "duplicate":
        records.append(records[0])
    elif corruption == "metadata":
        records[0]["seed"] = 999
    elif corruption == "probabilities":
        records[0]["before_probabilities"] = [0] * 8
    elif corruption == "target":
        records[0]["target_basis_indices"] = [0]
    elif corruption == "stored_probability":
        records[0]["before_p_target"] = 0.9
    elif corruption == "reference":
        manifest["instances"][0]["reference_objective"] += 1
    with pytest.raises(ValueError):
        sampling_rows(manifest, records)


def test_runner_creates_analysis_without_training_and_preserves_inputs(
    tmp_path, evidence, monkeypatch
):
    from scripts import run_sampling_allowance as runner
    from vqe_portfolio import training_diagnostics

    def forbidden(*args, **kwargs):
        pytest.fail("Sampling allowance analysis must never train a circuit.")

    monkeypatch.setattr(training_diagnostics, "train_circuit", forbidden)
    source = tmp_path / "source"
    source.mkdir()
    for name, data in zip(("manifest.json", "pairs.json"), evidence):
        (source / name).write_text(json.dumps(data))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "analysis"
    args = ["--diagnostic-dir", str(source), "--output-dir", str(output)]
    assert runner.main(args) == 0
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    for filename, count in (
        ("pairs.csv", 18),
        ("summary.csv", 3),
        ("baseline_trials.csv", 6),
        ("baseline_summary.csv", 3),
    ):
        with (output / filename).open() as stream:
            assert len(list(csv.DictReader(stream))) == count
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["training_performed"] and not manifest["sampling_performed"]
    assert manifest["draw_counts"] == [16, 64, 512]
    assert (output / "sampling_success.png").exists()
    assert "independent draws with replacement" in (output / "report.md").read_text()
    with pytest.raises(ValueError, match="nonempty"):
        runner.main(args)
