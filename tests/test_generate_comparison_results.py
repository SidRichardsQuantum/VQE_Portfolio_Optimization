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


def test_allocator_method_selection_accepts_the_legacy_alias_without_duplicates():
    from scripts.generate_comparison_results import parse_methods

    assert parse_methods("fractional-vqe") == {"variational-allocator"}
    assert parse_methods("fractional-vqe,variational-allocator") == {
        "variational-allocator"
    }
    assert "variational-allocator" in parse_methods("all")
    assert "fractional-vqe" not in parse_methods("all")


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
    assert len(rows) == 14
    assert len(runs) == 22
    assert rows[0]["objective_family"] == "fractional_simplex"
    assert rows[1]["solver_status"] == "optimal"
    assert float(rows[1]["mip_gap"]) < 1e-7
    assert rows[2]["method"] == "Classical quadratic integer (SCIP)"
    assert rows[2]["solver_status"] == "optimal"
    assert float(rows[2]["mip_gap"]) < 1e-7
    assert rows[3]["method"] == "Classical QP relaxation + rounding"
    assert rows[3]["type"] == "classical heuristic"
    assert rows[3]["seed_count"] == "1"
    assert rows[3]["lower_bound"] == "" and rows[3]["mip_gap"] == ""
    assert rows[4]["seed_count"] == "2"
    assert [
        r["seed"]
        for r in runs
        if r["method"] == "Classical variable neighborhood search"
    ] == ["7", "8"]
    assert rows[-3]["method"] == "Classical projected gradient descent"
    assert rows[-2]["method"] == "Classical Frank-Wolfe with away steps"
    assert rows[-2]["objective_family"] == "fractional_simplex"
    assert rows[-2]["seed_count"] == "1"
    assert rows[-1]["method"] == "Classical parallel tempering"
    assert rows[-1]["seed_count"] == "2"
    assert all(
        r["selection_or_weights"].count("1") == 3
        for r in runs
        if r["objective_family"] == "binary_qubo"
    )


def test_qp_rounding_can_be_selected_as_a_single_deterministic_binary_method(tmp_path):
    import pytest

    pytest.importorskip("cvxpy")
    pytest.importorskip("osqp")
    output, trials = tmp_path / "summary.csv", tmp_path / "trials.csv"
    assert (
        _load_script_main()(
            [
                "--methods",
                "qp-round",
                "--asset-counts",
                "4",
                "--k",
                "2",
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
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    with trials.open() as stream:
        runs = list(csv.DictReader(stream))
    assert len(rows) == len(runs) == 1
    assert rows[0]["method"] == "Classical QP relaxation + rounding"
    assert rows[0]["objective_family"] == "binary_qubo"
    assert rows[0]["feasible_rate"] == "1.000000"
    assert rows[0]["lower_bound"] == rows[0]["mip_gap"] == ""
    assert runs[0]["seed"] == "-"
    assert "relaxation_status=optimal" in runs[0]["notes"]


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
    monkeypatch.setattr(benchmark, "parallel_tempering_binary_baseline", failed)
    monkeypatch.setattr(benchmark, "tabu_search_binary_baseline", failed)
    monkeypatch.setattr(benchmark, "memetic_genetic_binary_baseline", failed)
    monkeypatch.setattr(benchmark, "iterated_local_search_binary_baseline", failed)
    monkeypatch.setattr(benchmark, "grasp_binary_baseline", failed)
    monkeypatch.setattr(benchmark, "relaxation_rounding_binary_baseline", failed)
    monkeypatch.setattr(
        benchmark, "variable_neighborhood_search_binary_baseline", failed
    )
    monkeypatch.setattr(benchmark, "projected_gradient_descent_baseline", failed)
    monkeypatch.setattr(benchmark, "frank_wolfe_baseline", failed)
    rows, trials = [], []
    benchmark.add_solver_rows(rows, trials, "test", np.ones(3), np.eye(3), 1, 2, [0, 1])
    assert len(rows) == 14 and len(trials) == 22
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
    monkeypatch.setattr(comparison, "relaxation_rounding_binary_baseline", unexpected)
    monkeypatch.setattr(comparison, "mixed_integer_binary_baseline", unexpected)
    monkeypatch.setattr(comparison, "scip_binary_baseline", unexpected)
    monkeypatch.setattr(comparison, "local_search_binary_baseline", unexpected)
    summary, trials = tmp_path / "summary.csv", tmp_path / "trials.csv"
    assert (
        _load_script_main()(
            [
                "--methods",
                "anneal,tabu,memetic,ils,grasp,vns,pgd,frank-wolfe,parallel-tempering",
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
        "Classical memetic genetic algorithm",
        "Classical iterated local search",
        "Classical GRASP",
        "Classical variable neighborhood search",
        "Classical projected gradient descent",
        "Classical Frank-Wolfe with away steps",
        "Classical parallel tempering",
    ]
    assert all(row["type"] == "classical heuristic" for row in rows[:6] + rows[-1:])
    assert all(row["type"] == "classical" for row in rows[6:8])
    assert all(row["feasible_rate"] == "1.000000" for row in rows)
    assert [run["seed"] for run in runs] == ["7", "8"] * 6 + ["-", "-", "7", "8"]
    assert all('"exchange_attempts"' in row["notes"] for row in runs[-2:])
    assert all(
        run["selection_or_weights"].count("1") == 2
        for run in runs
        if run["objective_family"] == "binary_qubo"
    )
