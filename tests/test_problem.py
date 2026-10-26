import itertools
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pennylane as qml
import pytest

import scripts.generate_comparison_results as benchmark
from vqe_portfolio.benchmark import CandidateTrial, comparison_rows
from vqe_portfolio.binary import build_ising_hamiltonian
from vqe_portfolio.classical import ClassicalResult
from vqe_portfolio.problem import PortfolioProblem
from vqe_portfolio.types import BinaryVQEConfig, FractionalVQEConfig, QAOAConfig


def test_frozen_snapshot_and_fingerprints(toy_problem):
    mu, sigma = toy_problem
    problem = PortfolioProblem(mu, sigma, 4, "binary_qubo", 2)
    identity = problem.problem_id
    mu[0] = 9
    sigma[:] = 0
    assert problem.mu[0] == 0.12
    with pytest.raises(ValueError):
        problem.mu[0] = 3
    with pytest.raises(ValueError):
        problem.Sigma.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        problem.lam = 9
    replay = PortfolioProblem.from_dict(json.loads(json.dumps(problem.to_dict())))
    assert replay.problem_id == identity
    for lam, family, k in [
        (5, "binary_qubo", 2),
        (4, "binary_qubo", 1),
        (4, "fractional_simplex", None),
    ]:
        assert (
            PortfolioProblem(problem.mu, problem.Sigma, lam, family, k).problem_id
            != identity
        )
    assert (
        PortfolioProblem(
            problem.mu + 0.01, problem.Sigma, 4, "binary_qubo", 2
        ).problem_id
        != identity
    )


@pytest.mark.parametrize(
    "candidate,reason",
    [
        ([1, 0, 0], "binary_or_cardinality_violation"),
        ([1, 0.5, 0.5], "binary_or_cardinality_violation"),
        ([1, 1], "wrong_shape"),
        ([1, np.nan, 0], "nonfinite_candidate"),
        ([1, np.inf, 0], "nonfinite_candidate"),
        (None, "missing_candidate"),
        ([1 + 1j, 1, 0], "complex_candidate"),
    ],
)
def test_invalid_binary_not_scored(toy_problem, candidate, reason):
    result = PortfolioProblem(*toy_problem, 4, "binary_qubo", 2).evaluate(
        candidate, alpha=7
    )
    assert not result.feasible
    assert result.objective is None
    assert result.reason == reason


def test_objective_track_and_penalty_separation(toy_problem):
    mu, sigma = toy_problem
    binary = PortfolioProblem(mu, sigma, 4, "binary_qubo", 2)
    fractional = PortfolioProblem(mu, sigma, 4, "fractional_simplex")
    x = np.array([1, 0, 1])
    result = binary.evaluate(x, alpha=100)
    assert result.objective == pytest.approx(0.49)
    assert result.penalized_energy == result.objective
    assert result.ret == pytest.approx(0.095)
    assert result.risk == pytest.approx(np.sqrt(0.17 / 4))
    assert fractional.evaluate(x / 2).objective == pytest.approx(0.075)
    assert binary.evaluate(x / 2).objective is None
    assert fractional.evaluate(x).objective is None
    invalid = binary.evaluate([1, 0, 0], alpha=7)
    assert invalid.objective is None
    assert invalid.penalized_energy == pytest.approx(7.28)


def test_simplex_validation_does_not_repair(toy_problem):
    problem = PortfolioProblem(*toy_problem, 4, "fractional_simplex")
    for w in [[0.2, 0.3, 0.4], [-0.1, 0.5, 0.6], [1, np.nan, 0], [0.5, 0.5 + 2e-8, 0]]:
        assert problem.evaluate(w).objective is None
    w = np.array([-0.5e-8, 0.5, 0.5 + 0.5e-8])
    original = w.copy()
    result = problem.evaluate(w)
    assert result.feasible
    np.testing.assert_array_equal(w, original)
    assert result.objective == pytest.approx(4 * w @ problem.Sigma @ w - problem.mu @ w)


@pytest.mark.parametrize("alpha", [0, 2, 9])
def test_hamiltonian_matches_shared_energy_on_every_bitstring(toy_problem, alpha):
    mu, sigma = toy_problem
    problem = PortfolioProblem(mu, sigma, 4, "binary_qubo", 2)
    diagonal = np.diag(
        qml.matrix(build_ising_hamiltonian(mu, sigma, 4, alpha, 2), wire_order=range(3))
    ).real
    for index, bits in enumerate(itertools.product([0, 1], repeat=3)):
        result = problem.evaluate(bits, alpha=alpha)
        assert result.penalized_energy == pytest.approx(diagonal[index], abs=1e-12)
        assert result.feasible == (sum(bits) == 2)


def test_all_adapters_share_problem_and_ignore_solver_scores(monkeypatch, toy_problem):
    mu, sigma = toy_problem
    x, w = np.array([1, 0, 1]), np.array([0.5, 0, 0.5])

    def old_binary(*args, **kwargs):
        return x, -999

    def old_fractional(*args, **kwargs):
        return w, -999

    for name in [
        "exhaustive_binary_baseline",
        "top_return_baseline",
        "minimum_variance_binary_baseline",
    ]:
        monkeypatch.setattr(benchmark, name, old_binary)
    for name in ["equal_weight_baseline", "exact_long_only_markowitz_baseline"]:
        monkeypatch.setattr(benchmark, name, old_fractional)

    def strong_binary(*args, **kwargs):
        return ClassicalResult(x, -999, "optimal")

    def strong_fractional(*args, **kwargs):
        return ClassicalResult(w, -999, "optimal")

    monkeypatch.setattr(benchmark, "mixed_integer_binary_baseline", strong_binary)
    monkeypatch.setattr(benchmark, "local_search_binary_baseline", strong_binary)
    monkeypatch.setattr(benchmark, "convex_markowitz_baseline", strong_fractional)

    def quantum_binary(mu_input, sigma_input, cfg):
        assert not mu_input.flags.writeable and not sigma_input.flags.writeable
        return SimpleNamespace(
            x_best_feasible=x, x_topk=x, sample_counts={"101": 1, "000": 3}
        )

    def quantum_fractional(*args, **kwargs):
        return SimpleNamespace(weights=w)

    monkeypatch.setattr(benchmark, "run_binary_vqe", quantum_binary)
    monkeypatch.setattr(benchmark, "run_qaoa", quantum_binary)
    monkeypatch.setattr(benchmark, "run_fractional_vqe", quantum_fractional)
    rows, trials = [], []
    benchmark.add_classical_rows(rows, trials, "test", mu, sigma, 4, 5, 2)
    benchmark.add_solver_rows(rows, trials, "test", mu, sigma, 4, 2, [0])
    benchmark.add_binary_vqe_row(
        rows, trials, "test", mu, sigma, BinaryVQEConfig(lam=4, k=2), [0]
    )
    for mixer in ["x", "xy"]:
        benchmark.add_qaoa_row(
            rows, trials, "test", mu, sigma, QAOAConfig(lam=4, k=2, mixer=mixer), [0]
        )
    benchmark.add_fractional_vqe_row(
        rows, trials, "test", mu, sigma, FractionalVQEConfig(lam=4), [0]
    )
    for family, expected in [("binary_qubo", 0.49), ("fractional_simplex", 0.075)]:
        group = [r for r in rows if r["objective_family"] == family]
        assert len({r["problem_id"] for r in group}) == 1
        assert all(float(r["best_objective"]) == pytest.approx(expected) for r in group)
        assert all(float(r["feasible_rate"]) == 1 for r in group)
    quantum_trials = [r for r in trials if r["type"] == "quantum" and r["k"]]
    assert all(float(r["sample_feasible_rate"]) == 0.25 for r in quantum_trials)
    assert len({r["problem_id"] for r in rows}) == 2


def test_quantum_invalid_candidate_not_rounded_and_failures_not_dropped(
    monkeypatch, toy_problem
):
    def solve(mu, sigma, cfg):
        x = np.array([1, 0, 1]) if cfg.seed == 0 else np.array([1.1, 0, 0.9])
        return SimpleNamespace(x_best_feasible=x, sample_counts={"101": 1})

    monkeypatch.setattr(benchmark, "run_binary_vqe", solve)
    rows, trials = [], []
    benchmark.add_binary_vqe_row(
        rows, trials, "test", *toy_problem, BinaryVQEConfig(lam=4, k=2), [0, 1]
    )
    assert float(rows[0]["feasible_rate"]) == 0.5
    assert rows[0]["seed_count"] == "2"
    assert float(rows[0]["best_objective"]) == pytest.approx(0.49)
    assert trials[1]["objective"] == ""
    assert trials[1]["validation_error"] == "binary_or_cardinality_violation"


def test_all_invalid_trials_keep_metadata(toy_problem):
    problem = PortfolioProblem(*toy_problem, 4, "fractional_simplex")
    summary, trials = comparison_rows(
        problem,
        [CandidateTrial(np.zeros(3), 2), CandidateTrial(None, 4)],
        dataset="test",
        method="broken",
        method_type="quantum",
    )
    assert summary["best_objective"] == ""
    assert summary["seed_count"] == "2"
    assert float(summary["mean_runtime_seconds"]) == 3
    assert all(r["validation_error"] for r in trials)


def test_generated_snapshot_replays_every_row(tmp_path):
    summary, trials = tmp_path / "summary.csv", tmp_path / "trials.csv"
    benchmark.main(
        [
            "--methods",
            "classical",
            "--asset-counts",
            "3",
            "--output",
            str(summary),
            "--trials-output",
            str(trials),
        ]
    )
    snapshots = json.loads(summary.with_suffix(".problems.json").read_text())
    import csv

    for row in csv.DictReader(trials.open()):
        problem = PortfolioProblem.from_dict(snapshots[row["problem_id"]])
        assert problem.problem_id == row["problem_id"]
        raw = row["selection_or_weights"]
        candidate = (
            [int(bit) for bit in raw]
            if problem.k is not None
            else np.fromstring(raw[1:-1], sep=" ")
        )
        result = problem.evaluate(candidate)
        assert result.objective == pytest.approx(float(row["objective"]), abs=1e-11)


def test_real_data_adapter_uses_independent_evaluator(monkeypatch, toy_problem):
    import vqe_portfolio.comparison as real

    mu, sigma = toy_problem
    monkeypatch.setattr(
        real,
        "get_stock_data",
        lambda *a, **kw: (
            SimpleNamespace(values=mu),
            SimpleNamespace(values=sigma),
            "prices",
        ),
    )

    def binary(*args, **kwargs):
        return SimpleNamespace(
            x_best_feasible=np.array([1.1, 0, 0.9]), sample_counts={"101": 1}
        )

    def fractional(*args, **kwargs):
        return SimpleNamespace(weights=np.zeros(3))

    monkeypatch.setattr(real, "run_binary_vqe", binary)
    monkeypatch.setattr(real, "run_qaoa", binary)
    monkeypatch.setattr(real, "run_fractional_vqe", fractional)
    rows, prices = real.real_data_comparison_rows(
        ["A", "B", "C"], start="2024-01-01", end="2025-01-01", lam=4, alpha=5, k=2
    )
    assert prices == "prices"
    quantum = [r for r in rows if r["type"] == "quantum"]
    assert len(quantum) == 4
    assert all(r["objective"] == "" and r["validation_error"] for r in quantum)
    assert len({r["problem_id"] for r in rows}) == 2


@pytest.mark.parametrize("candidate", [[[1], [0, 1]], [1 + 1j, 0, 1]])
def test_reporting_preserves_malformed_candidate_as_failure(toy_problem, candidate):
    problem = PortfolioProblem(*toy_problem, 4, "binary_qubo", 2)
    summary, trials = comparison_rows(
        problem,
        [CandidateTrial(candidate, 1)],
        dataset="test",
        method="broken",
        method_type="quantum",
    )
    assert summary["best_objective"] == ""
    assert trials[0]["validation_error"]
    assert trials[0]["selection_or_weights"]
