import itertools
import json

import numpy as np
import pytest

from vqe_portfolio.problem import PortfolioProblem
from vqe_portfolio.research import validation as v


def test_reference_matches_independent_enumeration_and_simplex_solution():
    mu = np.array([0.1, 0.3, 0.2])
    sigma = np.diag([0.2, 0.3, 0.4])
    binary = PortfolioProblem(mu, sigma, 2, "binary_qubo", 2)
    x, value = v.exact_reference(binary)
    candidates = [
        np.array(bits) for bits in itertools.product((0, 1), repeat=3) if sum(bits) == 2
    ]
    optimum = min(2 * c @ sigma @ c - mu @ c for c in candidates)
    assert binary.evaluate(x).feasible
    assert value == pytest.approx(optimum)
    fractional = PortfolioProblem(np.zeros(3), np.eye(3), 1, "fractional_simplex")
    w, value = v.exact_reference(fractional)
    np.testing.assert_allclose(w, np.ones(3) / 3)
    assert value == pytest.approx(1 / 3)


def test_instances_are_independent_frozen_and_fractional_is_not_duplicated():
    config = v.ValidationConfig(
        asset_counts=(4,), instance_seeds=(10, 20), k_fractions=(0.25, 0.5)
    )
    instances = v.prepare_instances(config)
    assert len(instances) == 6  # two binary cardinalities + one fractional per seed
    assert len({i["problem_id"] for i in instances}) == 6
    repeat = v.prepare_instances(config)
    assert [i["snapshot"] for i in instances] == [i["snapshot"] for i in repeat]
    for instance in instances:
        problem = PortfolioProblem.from_dict(instance["snapshot"])
        assert problem.problem_id == instance["problem_id"]
        assert np.linalg.eigvalsh(problem.Sigma).min() > 0
        assert problem.evaluate(instance["reference_candidate"]).feasible


def test_evaluation_does_not_trust_solver_objective_or_invalid_candidate(monkeypatch):
    config = v.ValidationConfig(
        asset_counts=(3,), instance_seeds=(1,), methods=("exact-binary",)
    )
    instance = v.prepare_instances(config)[0]
    monkeypatch.setattr(
        v,
        "solve",
        lambda *args: (np.array([np.nan, 0, 0]), {"solver_status": "optimal"}),
    )
    row = v.evaluate_trial("test", instance, "exact-binary", None, config)
    assert not row["feasible"] and not row["target_hit"]
    assert row["normalised_gap"] is None
    assert row["validation_error"] == "nonfinite_candidate"
    json.dumps(row, allow_nan=False)
    monkeypatch.setattr(
        v, "solve", lambda *args: (["bad"] * 3, {"lower_bound": float("inf")})
    )
    invalid = v.evaluate_trial("invalid", instance, "exact-binary", None, config)
    assert not invalid["feasible"]
    assert invalid["lower_bound"] is None and "nonfinite_lower_bound" in invalid
    json.dumps(invalid, allow_nan=False)

    def failed(*args):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(v, "solve", failed)
    row = v.evaluate_trial("test", instance, "exact-binary", None, config)
    assert row["error"] == "RuntimeError: backend failed"
    assert not row["feasible"] and not row["target_hit"]
    assert row["runtime_seconds"] >= 0


def test_negative_optimum_uses_positive_scale_and_flags_contradiction():
    config = v.ValidationConfig(
        asset_counts=(3,), instance_seeds=(1,), methods=("exact-binary",)
    )
    instance = v.prepare_instances(config)[0]
    problem = PortfolioProblem(
        np.array([1, 2, 3]), np.eye(3) * 0.01, 1, "binary_qubo", 1
    )
    instance.update(
        snapshot=problem.to_dict(),
        problem_id=problem.problem_id,
        reference_objective=-2.99,
        objective_scale=v.objective_scale(problem),
    )
    row = v.evaluate_trial("test", instance, "exact-binary", None, config)
    assert row["normalised_gap"] == pytest.approx(0)
    assert row["target_hit"]
    instance["reference_objective"] += instance["objective_scale"] * 0.1
    row = v.evaluate_trial("test", instance, "exact-binary", None, config)
    assert row["reference_contradiction"]
    assert not row["target_hit"]


def test_hamiltonian_matches_original_penalised_objective_for_every_bitstring():
    import pennylane as qml

    from vqe_portfolio.binary import build_ising_hamiltonian

    mu, sigma = v.synthetic_instance(4, 9)
    hamiltonian = build_ising_hamiltonian(mu, sigma, 4, 5, 2)
    expected = []
    for bits in itertools.product((0, 1), repeat=4):
        x = np.array(bits)
        expected.append(4 * x @ sigma @ x - mu @ x + 5 * (x.sum() - 2) ** 2)
    matrix = qml.matrix(hamiltonian, wire_order=range(4))
    np.testing.assert_allclose(matrix, np.diag(expected), atol=1e-12)


def test_topk_fallback_is_explicit_and_still_independently_scored(monkeypatch):
    from types import SimpleNamespace

    config = v.ValidationConfig(
        asset_counts=(3,), instance_seeds=(1,), methods=("binary-vqe",)
    )
    instance = v.prepare_instances(config)[0]
    monkeypatch.setattr(
        v,
        "run_binary_vqe",
        lambda *args: SimpleNamespace(
            x_best_feasible=None, x_topk=np.array([1, 0, 0]), sample_counts={"000": 512}
        ),
    )
    row = v.evaluate_trial("test", instance, "binary-vqe", 0, config)
    assert row["fallback_used"] and row["sample_feasible_rate"] == 0
    assert row["feasible"]
    problem = PortfolioProblem.from_dict(instance["snapshot"])
    assert row["objective"] == pytest.approx(problem.evaluate([1, 0, 0]).objective)


def test_summary_weights_instances_and_preserves_failed_runs():
    base = {
        "n_assets": 4,
        "k": 2,
        "objective_family": "binary_qubo",
        "method": "test",
        "runtime_seconds": 1,
        "error": "",
        "reference_contradiction": False,
    }
    rows = [
        dict(base, problem_id="a", feasible=True, normalised_gap=0.2, target_hit=False),
        dict(
            base,
            problem_id="a",
            feasible=False,
            normalised_gap=None,
            target_hit=False,
            error="failed",
        ),
        dict(base, problem_id="b", feasible=True, normalised_gap=0, target_hit=True),
        dict(
            base, problem_id="c", feasible=False, normalised_gap=None, target_hit=False
        ),
    ]
    summary = v.summarise_trials(rows)[0]
    assert summary["instance_count"] == 3 and summary["trial_count"] == 4
    assert summary["quality_instance_count"] == 2
    assert summary["mean_normalised_gap_feasible"] == pytest.approx(0.1)
    assert summary["feasible_rate"] == pytest.approx(0.5)
    assert summary["target_hit_rate"] == pytest.approx(1 / 3)
    assert summary["error_count"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"asset_counts": (17,)},
        {"steps": 0},
        {"shots": 0},
        {"instance_seeds": (1, 1)},
        {"solver_seeds": (-1,)},
        {"k_fractions": (1.0,)},
        {"methods": ("unknown",)},
        {"target_gap": float("nan")},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        v.ValidationConfig(**kwargs)


def test_quantum_adapters_return_valid_small_portfolios():
    config = v.ValidationConfig(
        asset_counts=(3,),
        instance_seeds=(11,),
        solver_seeds=(0,),
        steps=1,
        shots=32,
        methods=("binary-vqe", "qaoa-x", "qaoa-xy", "fractional-vqe"),
    )
    for key, instance, method, seed in v.trial_plan(
        v.prepare_instances(config), config
    ):
        row = v.evaluate_trial(key, instance, method, seed, config)
        assert row["error"] == ""
        assert row["feasible"]
        assert row["normalised_gap"] >= -1e-8
        if method != "fractional-vqe":
            assert 0 <= row["sample_feasible_rate"] <= 1


def test_random_baseline_matches_independent_uniform_subset_sampling():
    config = v.ValidationConfig(
        asset_counts=(4,), instance_seeds=(10,), methods=("random-k",), shots=7
    )
    instance = v.prepare_instances(config)[0]
    problem = PortfolioProblem.from_dict(instance["snapshot"])
    candidate, _ = v.solve(problem, "random-k", 123, config)
    rng = np.random.default_rng(123)
    scores = []
    for _ in range(7):
        x = np.zeros(4, dtype=int)
        x[rng.choice(4, 2, replace=False)] = 1
        scores.append(problem.evaluate(x).objective)
    assert problem.evaluate(candidate).objective == pytest.approx(min(scores))


def test_checkpoint_resume_and_changed_config_refusal(tmp_path, monkeypatch):
    from scripts import run_small_instance_validation as runner

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "asset_counts": [3],
                "instance_seeds": [1],
                "solver_seeds": [0],
                "methods": ["exact-binary", "exact-fractional"],
            }
        )
    )
    output = tmp_path / "results"
    args = ["--config", str(config), "--output-dir", str(output)]
    assert runner.main(args) == 0
    checkpoint = json.loads((output / "trials.json").read_text())
    assert len(checkpoint) == 2
    # Emulate interruption after one persisted trial: only the missing one reruns.
    (output / "trials.json").write_text(json.dumps(checkpoint[:1]))
    original = runner.evaluate_trial
    calls = []

    def record(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(runner, "evaluate_trial", record)
    assert runner.main(args + ["--resume"]) == 0
    assert calls == [checkpoint[1]["trial_id"]]
    assert json.loads((output / "trials.json").read_text())[0] == checkpoint[0]
    calls.clear()
    assert runner.main(args + ["--resume"]) == 0
    assert calls == []
    with pytest.raises(ValueError, match="nonempty"):
        runner.main(args)
    settings = json.loads(config.read_text())
    settings["steps"] = 99
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="Resume refused"):
        runner.main(args + ["--resume"])
