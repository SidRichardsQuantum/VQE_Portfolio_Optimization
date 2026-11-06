import time

import numpy as np
import pytest

from vqe_portfolio import execution
from vqe_portfolio.execution import CandidateSnapshot, _PhaseTimings, run_budgeted
from vqe_portfolio.problem import PortfolioProblem
from vqe_portfolio.research.budget import summarise_budget_trials
from vqe_portfolio.types import (
    BinaryVQEConfig,
    FractionalVQEConfig,
    FVQEConfig,
    QAOAConfig,
)


def worker_incumbent_then_block(
    connection, snapshot, method, config, seed, started, budget
):
    connection.send(
        (
            "candidate",
            CandidateSnapshot(
                time.perf_counter() - started,
                [1, 0, 0],
                -999,
                {"solver_diagnostics": {"local_proposals": 2}},
            ),
        )
    )
    time.sleep(budget + 1)


def worker_late(connection, snapshot, method, config, seed, started, budget):
    time.sleep(budget + 0.1)
    connection.send(("candidate", CandidateSnapshot(0, [1, 0, 0], -999)))
    connection.send(("done", ("completed", "")))


def worker_invalid_then_error(
    connection, snapshot, method, config, seed, started, budget
):
    connection.send(("candidate", CandidateSnapshot(0, [1, 1, 0], -999)))
    connection.send(("candidate", CandidateSnapshot(0, [1, 0, 0], -999)))
    connection.send(("done", ("error", "Backend failure")))


@pytest.fixture
def problem():
    return PortfolioProblem([0.1, 0.2, 0.3], np.eye(3) * 0.1, 1.0, "binary_qubo", 1)


def test_timeout_retains_only_independently_scored_incumbent(monkeypatch, problem):
    monkeypatch.setattr(execution, "_budget_worker", worker_incumbent_then_block)
    result = run_budgeted(problem, "random-k", 5)
    assert result.status == "time_limit"
    assert result.candidate == [1, 0, 0]
    assert result.objective == pytest.approx(0)
    assert 0 < result.candidate_seconds <= 5
    assert result.elapsed_seconds < 6.5
    assert result.overrun_seconds >= 0
    assert result.solver_diagnostics == {}
    assert result.incumbents[0].details["solver_diagnostics"] == {"local_proposals": 2}


def test_late_candidate_is_never_scored(monkeypatch, problem):
    monkeypatch.setattr(execution, "_budget_worker", worker_late)
    result = run_budgeted(problem, "random-k", 0.1)
    assert result.status == "time_limit"
    assert result.candidate is None and result.objective is None
    assert result.incumbents == ()


def test_error_keeps_incumbent_and_rejects_invalid_output(monkeypatch, problem):
    monkeypatch.setattr(execution, "_budget_worker", worker_invalid_then_error)
    result = run_budgeted(problem, "random-k", 10)
    assert result.status == "error"
    assert result.error == "Backend failure"
    assert result.invalid_candidates == 1
    assert result.objective == pytest.approx(0)


@pytest.mark.parametrize(
    "method",
    [
        "binary-vqe",
        "f-vqe",
        "qaoa-x",
        "qaoa-xy",
        "qaoa-xy-untrained",
        "qaoa-xy-trained",
        "random-k",
        "fractional-vqe",
        "variational-allocator",
        "swap",
        "anneal",
        "parallel-tempering",
        "tabu",
        "memetic",
        "ils",
        "grasp",
        "highs",
        "scip",
        "osqp",
        "qp-round",
        "vns",
        "pgd",
        "frank-wolfe",
    ],
)
def test_real_budget_workers_return_feasible_portfolios(method, problem):
    if method in {"highs", "swap"}:
        pytest.importorskip("scipy")
    if method == "scip":
        pytest.importorskip("pyscipopt")
    if method in {"osqp", "qp-round"}:
        pytest.importorskip("cvxpy")
        pytest.importorskip("osqp")
    config = None
    if method == "binary-vqe":
        config = BinaryVQEConfig(
            k=1, lam=1, steps=1, log_every=1, shots_sample=8, alpha="auto"
        )
    elif method == "f-vqe":
        config = FVQEConfig(
            k=1, lam=1, steps=1, log_every=1, shots_train=16, shots_sample=8
        )
    elif method.startswith("qaoa-"):
        xy = method.startswith("qaoa-xy")
        config = QAOAConfig(
            k=1,
            lam=1,
            p=1,
            steps=0 if method.endswith("-untrained") else 1,
            log_every=1,
            shots_sample=8,
            mixer="xy" if xy else "x",
            xy_initial_state="dicke" if xy else "basis",
        )
    elif method in {"fractional-vqe", "variational-allocator"}:
        config = FractionalVQEConfig(lam=1, steps=1, log_every=1)
    if method in {
        "fractional-vqe",
        "variational-allocator",
        "osqp",
        "pgd",
        "frank-wolfe",
    }:
        problem = PortfolioProblem(problem.mu, problem.Sigma, 1.0, "fractional_simplex")
    result = run_budgeted(problem, method, 15, config=config)
    assert result.status == "completed", result.error
    if method == "parallel-tempering":
        assert result.solver_diagnostics["local_proposals"] == 16 * 8 * 1000
        assert result.solver_diagnostics["completed_restarts"] == 16
        assert result.solver_diagnostics["exchange_attempts"] > 0
    assert result.timings["startup_seconds"] > 0
    assert result.timings["setup_seconds"] > 0
    assert result.timings["parent_evaluation_seconds"] > 0
    assert result.timings["mean_delivery_seconds"] >= 0
    if method.endswith("-untrained"):
        assert result.timings["optimization_seconds"] == 0
        assert result.solver_status == "exhausted_feasible_space"
    if method == "random-k":
        assert result.solver_status == "exhausted_feasible_space"
    assert problem.evaluate(result.candidate).feasible
    assert result.objective == pytest.approx(
        problem.evaluate(result.candidate).objective
    )
    assert all(s.elapsed_seconds <= 15 for s in result.incumbents)
    assert all(
        a.objective > b.objective
        for a, b in zip(result.incumbents, result.incumbents[1:])
    )


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_budgets(problem, seconds):
    with pytest.raises(ValueError, match="budget_seconds"):
        run_budgeted(problem, "swap", seconds)


def test_mismatched_problem_and_solver_configs_are_rejected(problem):
    with pytest.raises(ValueError, match="objective family"):
        run_budgeted(problem, "osqp", 1)
    with pytest.raises(ValueError, match="objective family"):
        run_budgeted(problem, "frank-wolfe", 1)
    with pytest.raises(ValueError, match="match the prepared problem"):
        run_budgeted(problem, "binary-vqe", 1, config=BinaryVQEConfig(k=2))
    with pytest.raises(ValueError, match="mixer disagree"):
        run_budgeted(problem, "qaoa-xy", 1, config=QAOAConfig(k=1, lam=1))
    with pytest.raises(ValueError, match="steps=0"):
        run_budgeted(
            problem, "qaoa-xy-untrained", 1, config=QAOAConfig(k=1, lam=1, mixer="xy")
        )


def test_phase_accounting_retains_interrupted_work_and_excludes_shutdown():
    timer = _PhaseTimings()
    timer.mark("setup", 0.2)
    timer.mark("optimization", 0.5)
    timer.mark("candidate_processing", 1.0)
    timer.mark("optimization", 1.3)
    totals = timer.finish(2.0)
    assert totals == pytest.approx(
        {
            "startup_seconds": 0.2,
            "setup_seconds": 0.3,
            "optimization_seconds": 1.2,
            "candidate_processing_seconds": 0.3,
        }
    )
    timer.mark("finished", 2.0)
    assert timer.finish(5.0) == pytest.approx(totals)
    with pytest.raises(ValueError, match="Nonmonotonic"):
        timer.mark("setup", 1.0)


def test_parent_validation_must_finish_before_deadline(monkeypatch, problem):
    monkeypatch.setattr(execution, "_budget_worker", worker_invalid_then_error)
    evaluate = PortfolioProblem.evaluate

    def slow_evaluation(self, candidate):
        time.sleep(5)
        return evaluate(self, candidate)

    monkeypatch.setattr(PortfolioProblem, "evaluate", slow_evaluation)
    result = run_budgeted(problem, "random-k", 5)
    assert result.candidate is None
    assert result.late_candidates == 1
    assert result.timings["parent_evaluation_seconds"] >= 5


def test_t90_includes_failures_instead_of_success_only_timing():
    base = dict(
        n_assets=4,
        k=2,
        method="qaoa-x",
        budget_seconds=10,
        problem_id="one",
        normalised_gap=0.1,
        feasible=True,
        status="completed",
        reference_contradiction=False,
        elapsed_seconds=1,
        overrun_seconds=0,
    )
    trials = [
        {**base, "target_hit": i < 8, "time_to_target_seconds": 1 if i < 8 else None}
        for i in range(10)
    ]
    summary, reliability = summarise_budget_trials(trials)
    assert reliability[0]["t90_seconds"] is None
    assert summary[0]["target_hit_rate"] == 0.8
    trials[8].update(target_hit=True, time_to_target_seconds=3)
    assert summarise_budget_trials(trials)[1][0]["t90_seconds"] == 3
    trials[9].update(feasible=False, normalised_gap=None, status="error")
    summary = summarise_budget_trials(trials)[0][0]
    assert summary["feasible_rate"] == 0.9 and summary["error_count"] == 1
