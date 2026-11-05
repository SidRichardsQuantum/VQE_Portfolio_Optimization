"""End-to-end execution budgets and independently evaluated incumbents.

Spawned workers isolate RNG state and allow termination of noncooperative native
solvers. Worker startup, encoding, training, candidate measurement, repair, and
evaluation all consume the budget. No reference optimum is sent to a worker.
"""

import multiprocessing as mp
import time
from dataclasses import dataclass, field

import numpy as np

from .problem import PortfolioProblem
from .types import BinaryVQEConfig, FractionalVQEConfig, QAOAConfig


BUDGET_METHODS = {
    "binary-vqe": "binary_qubo",
    "qaoa-x": "binary_qubo",
    "qaoa-xy": "binary_qubo",
    "highs": "binary_qubo",
    "swap": "binary_qubo",
    "random-k": "binary_qubo",
    "fractional-vqe": "fractional_simplex",
    "osqp": "fractional_simplex",
}


@dataclass(frozen=True)
class CandidateSnapshot:
    elapsed_seconds: float
    candidate: list[float]
    objective: float
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetResult:
    method: str
    problem_id: str
    budget_seconds: float
    elapsed_seconds: float
    status: str
    candidate: list[float] | None
    objective: float | None
    candidate_seconds: float | None
    incumbents: tuple[CandidateSnapshot, ...] = ()
    solver_status: str = ""
    error: str = ""
    invalid_candidates: int = 0
    late_candidates: int = 0
    worker_exitcode: int | None = None
    overrun_seconds: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "overrun_seconds",
            max(0.0, self.elapsed_seconds - self.budget_seconds),
        )


def _configuration(problem, method, config, seed):
    expected = {
        "binary-vqe": BinaryVQEConfig,
        "qaoa-x": QAOAConfig,
        "qaoa-xy": QAOAConfig,
        "fractional-vqe": FractionalVQEConfig,
    }.get(method)
    if expected is None:
        if config is not None:
            raise ValueError("Classical budget methods do not accept a quantum config")
        return None
    if config is None:
        common = {"lam": problem.lam, "seed": seed}
        if problem.k is not None:
            common["k"] = problem.k
        if method.startswith("qaoa-"):
            common["mixer"] = method.removeprefix("qaoa-")
        config = expected(**common)
    if not isinstance(config, expected):
        raise ValueError(f"{method} requires {expected.__name__}")
    if config.lam != problem.lam or (problem.k is not None and config.k != problem.k):
        raise ValueError("Solver configuration must match the prepared problem")
    if method.startswith("qaoa-") and config.mixer != method.removeprefix("qaoa-"):
        raise ValueError("Method and configured mixer disagree")
    return config


def _solve(problem, method, config, seed, deadline, publish):
    """Worker-side dispatch; all method-specific preparation is inside timing."""
    mu, sigma, lam, k = problem.mu, problem.Sigma, problem.lam, problem.k
    if method == "binary-vqe":
        from .binary import run_binary_vqe

        run_binary_vqe(mu, sigma, config, candidate_callback=publish)
    elif method.startswith("qaoa-"):
        from .qaoa import run_qaoa

        run_qaoa(mu, sigma, config, candidate_callback=publish)
    elif method == "fractional-vqe":
        from .fractional import run_fractional_vqe

        run_fractional_vqe(mu, sigma, config, candidate_callback=publish)
    elif method == "random-k":
        rng = np.random.default_rng(seed)
        while time.perf_counter() < deadline:
            candidate = np.zeros(len(mu), dtype=int)
            candidate[rng.choice(len(mu), k, replace=False)] = 1
            publish(candidate, {"source": "uniform_exact_k"})
    else:
        from .baselines.solvers import (
            convex_markowitz_baseline,
            local_search_binary_baseline,
            mixed_integer_binary_baseline,
        )

        if method == "highs":
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return "time_limit"
            result = mixed_integer_binary_baseline(
                mu, sigma, lam, k, time_limit=remaining
            )
        elif method == "osqp":
            result = convex_markowitz_baseline(mu, sigma, lam)
        else:
            result = local_search_binary_baseline(
                mu, sigma, lam, k, seed=seed, candidate_callback=publish
            )
        publish(result.candidate, {"source": method, "solver_status": result.status})
        return result.status
    return "completed"


def _budget_worker(connection, snapshot, method, config, seed, started, budget):
    try:
        problem = PortfolioProblem.from_dict(snapshot)
        best = float("inf")

        def publish(candidate, details):
            nonlocal best
            evaluated = problem.evaluate(candidate)
            elapsed = time.perf_counter() - started
            if not evaluated.feasible:
                connection.send(("invalid", None))
            elif evaluated.objective < best:
                # Serialization and delivery must also finish before the deadline;
                # the parent checks reception time independently.
                best = evaluated.objective
                connection.send(
                    (
                        "candidate",
                        CandidateSnapshot(
                            elapsed,
                            np.asarray(candidate, dtype=float).tolist(),
                            evaluated.objective,
                            details,
                        ),
                    )
                )

        status = _solve(problem, method, config, seed, started + budget, publish)
        connection.send(("done", (status, "")))
    except Exception as exc:
        connection.send(("done", ("error", f"{type(exc).__name__}: {exc}")))
    finally:
        connection.close()


def run_budgeted(
    problem: PortfolioProblem,
    method: str,
    budget_seconds: float,
    *,
    config=None,
    seed: int = 0,
) -> BudgetResult:
    """Return the best feasible portfolio received within a wall-clock budget.

    Use the usual ``if __name__ == '__main__'`` guard in Python scripts because
    workers use multiprocessing's portable ``spawn`` method. A timeout may have
    no candidate, including when startup alone consumes a very short budget.
    Candidate checkpoints use the solver's ``log_every`` and consume resources;
    freeze their cadence before comparing runs. Cleanup overruns are reported,
    and never extend candidate eligibility. Quantum result arrays are available
    from the ordinary solver APIs; this API returns common execution metadata.
    """
    if method not in BUDGET_METHODS:
        raise ValueError(f"Unknown budget method: {method}")
    if BUDGET_METHODS[method] != problem.objective_family:
        raise ValueError("Method does not match the problem objective family")
    if not np.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("budget_seconds must be finite and positive")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    config = _configuration(problem, method, config, seed)
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    incumbents = []
    invalid, late = 0, 0
    status, solver_status, error = "time_limit", "", ""
    started = time.perf_counter()
    worker = context.Process(
        target=_budget_worker,
        args=(
            send,
            problem.to_dict(),
            method,
            config,
            seed,
            started,
            budget_seconds,
        ),
    )
    try:
        worker.start()
        send.close()
        while True:
            remaining = started + budget_seconds - time.perf_counter()
            if remaining <= 0 or not receive.poll(remaining):
                break
            try:
                kind, payload = receive.recv()
            except EOFError:
                status, solver_status, error = (
                    "error",
                    "worker_exit",
                    "Worker exited without completion metadata",
                )
                break
            received = time.perf_counter() - started
            if kind == "candidate":
                if (
                    received > budget_seconds
                    or payload.elapsed_seconds > budget_seconds
                ):
                    late += 1
                    continue
                # Independently rescore the raw worker output; never trust its score.
                evaluation = problem.evaluate(payload.candidate)
                if not evaluation.feasible:
                    invalid += 1
                elif not incumbents or evaluation.objective < incumbents[-1].objective:
                    incumbents.append(
                        CandidateSnapshot(
                            received,
                            payload.candidate,
                            evaluation.objective,
                            payload.details,
                        )
                    )
            elif kind == "invalid":
                invalid += 1
            elif kind == "done":
                if received <= budget_seconds:
                    solver_status, error = payload
                    status = "error" if error else "completed"
                break
    except Exception as exc:
        status, solver_status, error = (
            "error",
            "execution_error",
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        if worker.pid is not None:
            if status != "time_limit":
                worker.join(timeout=0.05)
            if worker.is_alive():
                # Even successful workers can still be shutting down their runtime.
                worker.terminate()
            worker.join(timeout=0.25)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=0.25)
        receive.close()
        send.close()
    elapsed = time.perf_counter() - started
    best = incumbents[-1] if incumbents else None
    result = BudgetResult(
        method,
        problem.problem_id,
        float(budget_seconds),
        elapsed,
        status,
        best.candidate if best else None,
        best.objective if best else None,
        best.elapsed_seconds if best else None,
        tuple(incumbents),
        solver_status,
        error,
        invalid,
        late,
        worker.exitcode,
    )
    worker.close()
    return result
