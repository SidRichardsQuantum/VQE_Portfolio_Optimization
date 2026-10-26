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
    assert len(rows) == 3
    assert len(runs) == 4
    assert rows[0]["objective_family"] == "fractional_simplex"
    assert rows[1]["solver_status"] == "optimal"
    assert float(rows[1]["mip_gap"]) < 1e-7
    assert rows[2]["seed_count"] == "2"
    assert [r["seed"] for r in runs[-2:]] == ["7", "8"]
    assert all(r["selection_or_weights"].count("1") == 3 for r in runs[1:])


def test_strong_classical_failures_are_preserved(monkeypatch):
    import numpy as np

    import scripts.generate_comparison_results as benchmark
    from vqe_portfolio.classical import ClassicalResult

    def failed(*a, **kw):
        return ClassicalResult(None, None, "limit", lower_bound=-3.0)

    monkeypatch.setattr(benchmark, "convex_markowitz_baseline", failed)
    monkeypatch.setattr(benchmark, "mixed_integer_binary_baseline", failed)
    monkeypatch.setattr(benchmark, "local_search_binary_baseline", failed)
    rows, trials = [], []
    benchmark.add_solver_rows(rows, trials, "test", np.ones(3), np.eye(3), 1, 2, [0, 1])
    assert len(rows) == 3 and len(trials) == 4
    assert all(r["feasible_rate"] == "0.000000" for r in rows)
    assert all(r["best_objective"] == "" for r in rows)
    assert all(r["objective"] == "" for r in trials)
    assert rows[1]["lower_bound"] == "-3.0"

    assert all(set(r) <= set(benchmark.FIELDNAMES) for r in rows)
    assert all(r["lambda"] == "1" for r in rows)
