from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_script_main():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_comparison_results.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_comparison_results", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_generate_comparison_results_classical_smoke(tmp_path):
    main = _load_script_main()
    output = tmp_path / "summary.csv"
    trials_output = tmp_path / "trials.csv"

    rc = main(
        [
            "--output",
            str(output),
            "--trials-output",
            str(trials_output),
            "--asset-counts",
            "3",
            "--methods",
            "classical",
        ]
    )

    assert rc == 0
    with output.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with trials_output.open(newline="") as f:
        trial_rows = list(csv.DictReader(f))

    assert [row["method"] for row in rows] == [
        "Classical exhaustive search",
        "Classical top-return heuristic",
        "Classical minimum-variance subset",
        "Classical equal weight",
        "Classical exact Markowitz",
    ]
    assert rows[0]["n_assets"] == "3"
    assert rows[0]["k"] == "2"
    assert rows[3]["k"] == ""
    assert len(trial_rows) == len(rows)
    assert trial_rows[0]["seed"] == "-"


def test_strong_classical_benchmark(tmp_path):
    import pytest

    pytest.importorskip("cvxpy")
    pytest.importorskip("scipy.optimize")
    pytest.importorskip("pyscipopt")
    main = _load_script_main()
    output, trials = tmp_path / "summary.csv", tmp_path / "trials.csv"
    assert (
        main(
            [
                "--methods",
                "classical-solvers",
                "--asset-counts",
                "5",
                "--k",
                "3",
                "--seeds",
                "7,8",
                "--output",
                str(output),
                "--trials-output",
                str(trials),
            ]
        )
        == 0
    )
    with output.open() as f:
        rows = list(csv.DictReader(f))
    with trials.open() as f:
        runs = list(csv.DictReader(f))
    assert len(rows) == 6
    assert len(runs) == 9
    assert rows[0]["objective_family"] == "fractional_simplex"
    assert rows[1]["solver_status"] == "optimal"
    assert float(rows[1]["mip_gap"]) < 1e-7
    assert rows[2]["method"] == "Classical quadratic integer (SCIP)"
    assert rows[2]["solver_status"] == "optimal"
    assert float(rows[2]["mip_gap"]) < 1e-7
    assert rows[3]["seed_count"] == "2"
    assert [r["seed"] for r in runs[-2:]] == ["7", "8"]
    assert all(r["selection_or_weights"].count("1") == 3 for r in runs[1:])


def test_strong_classical_failures_are_preserved(monkeypatch):
    import numpy as np

    from vqe_portfolio.baselines import ClassicalResult
    from vqe_portfolio.research import comparison as benchmark

    def failed(*a, **kw):
        return ClassicalResult(None, None, "limit", lower_bound=-3.0)

    monkeypatch.setattr(benchmark, "convex_markowitz_baseline", failed)
    monkeypatch.setattr(benchmark, "mixed_integer_binary_baseline", failed)
    monkeypatch.setattr(benchmark, "scip_binary_baseline", failed)
    monkeypatch.setattr(benchmark, "local_search_binary_baseline", failed)
    monkeypatch.setattr(benchmark, "simulated_annealing_binary_baseline", failed)
    monkeypatch.setattr(benchmark, "tabu_search_binary_baseline", failed)
    rows, trials = [], []
    benchmark.add_solver_rows(rows, trials, "test", np.ones(3), np.eye(3), 1, 2, [0, 1])
    assert len(rows) == 6 and len(trials) == 9
    assert all(r["feasible_rate"] == "0.000000" for r in rows)
    assert all(r["best_objective"] == "" for r in rows)
    assert all(r["objective"] == "" for r in trials)
    assert float(rows[1]["lower_bound"]) == -3.0

    assert all(set(r) <= set(benchmark.FIELDNAMES) for r in rows)
    assert all(r["lambda"] == "1" for r in rows)


def test_numpy_search_methods_can_run_without_optional_solvers(tmp_path, monkeypatch):
    from vqe_portfolio.research import comparison

    def unexpected(*args, **kwargs):
        raise AssertionError("An unselected solver was called")

    monkeypatch.setattr(comparison, "convex_markowitz_baseline", unexpected)
    monkeypatch.setattr(comparison, "mixed_integer_binary_baseline", unexpected)
    monkeypatch.setattr(comparison, "scip_binary_baseline", unexpected)
    monkeypatch.setattr(comparison, "local_search_binary_baseline", unexpected)
    summary, trials = tmp_path / "summary.csv", tmp_path / "trials.csv"
    assert (
        _load_script_main()(
            [
                "--methods",
                "anneal,tabu",
                "--asset-counts",
                "4",
                "--k",
                "2",
                "--seeds",
                "7,8",
                "--local-restarts",
                "2",
                "--output",
                str(summary),
                "--trials-output",
                str(trials),
            ]
        )
        == 0
    )
    with summary.open() as f:
        rows = list(csv.DictReader(f))
    with trials.open() as f:
        runs = list(csv.DictReader(f))
    assert [row["method"] for row in rows] == [
        "Classical simulated annealing",
        "Classical tabu search",
    ]
    assert all(row["type"] == "classical heuristic" for row in rows)
    assert all(row["feasible_rate"] == "1.000000" for row in rows)
    assert [run["seed"] for run in runs] == ["7", "8", "7", "8"]
    assert all(run["selection_or_weights"].count("1") == 2 for run in runs)
