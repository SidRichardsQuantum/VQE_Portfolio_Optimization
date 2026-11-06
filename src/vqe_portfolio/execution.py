"""End-to-end execution budgets and independently evaluated incumbents.

Spawned workers isolate RNG state and allow termination of noncooperative native
solvers. Worker startup, encoding, training, candidate measurement, repair, and
evaluation all consume the budget. No reference optimum is sent to a worker.
"""

import multiprocessing as mp
import time
from dataclasses import dataclass, field
from math import comb

import numpy as np

from .problem import PortfolioProblem
from .types import BinaryVQEConfig, FractionalVQEConfig, FVQEConfig, QAOAConfig


BUDGET_METHODS = {
    "binary-vqe": "binary_qubo",
    "f-vqe": "binary_qubo",
    "qaoa-x": "binary_qubo",
    "qaoa-xy": "binary_qubo",
    "qaoa-xy-untrained": "binary_qubo",
    "qaoa-xy-trained": "binary_qubo",
    "highs": "binary_qubo",
    "scip": "binary_qubo",
    "qp-round": "binary_qubo",
    "swap": "binary_qubo",
    "anneal": "binary_qubo",
    "parallel-tempering": "binary_qubo",
    "tabu": "binary_qubo",
    "memetic": "binary_qubo",
    "ils": "binary_qubo",
    "grasp": "binary_qubo",
    "vns": "binary_qubo",
    "random-k": "binary_qubo",
    "fractional-vqe": "fractional_simplex",
    "variational-allocator": "fractional_simplex",
    "osqp": "fractional_simplex",
    "pgd": "fractional_simplex",
    "frank-wolfe": "fractional_simplex",
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
    timings: dict = field(default_factory=dict)
    solver_diagnostics: dict = field(default_factory=dict)
    overrun_seconds: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "overrun_seconds",
            max(0.0, self.elapsed_seconds - self.budget_seconds),
        )


class _PhaseTimings:
    """Worker-clock phase totals, including a censored active phase on timeout."""

    phases = ("startup", "setup", "optimization", "candidate_processing")

    def __init__(self):
        self.totals = {name: 0.0 for name in self.phases}
        self.active = "startup"
        self.last = 0.0

    def mark(self, phase, elapsed):
        if phase not in (*self.phases, "finished"):
            raise ValueError("Unknown worker timing phase")
        if not np.isfinite(elapsed) or elapsed < self.last:
            raise ValueError("Nonmonotonic worker timing")
        if self.active != "finished":
            self.totals[self.active] += elapsed - self.last
        self.active, self.last = phase, elapsed

    def finish(self, elapsed):
        self.mark(self.active, max(self.last, elapsed))
        return {f"{key}_seconds": value for key, value in self.totals.items()}


def _configuration(problem, method, config, seed):
    expected = {
        "binary-vqe": BinaryVQEConfig,
        "f-vqe": FVQEConfig,
        "qaoa-x": QAOAConfig,
        "qaoa-xy": QAOAConfig,
        "qaoa-xy-untrained": QAOAConfig,
        "qaoa-xy-trained": QAOAConfig,
        "fractional-vqe": FractionalVQEConfig,
        "variational-allocator": FractionalVQEConfig,
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
            common["mixer"] = "x" if method == "qaoa-x" else "xy"
        if method == "qaoa-xy-untrained":
            common["steps"] = 0
        config = expected(**common)
    if not isinstance(config, expected):
        raise ValueError(f"{method} requires {expected.__name__}")
    if config.lam != problem.lam or (problem.k is not None and config.k != problem.k):
        raise ValueError("Solver configuration must match the prepared problem")
    mixer = "x" if method == "qaoa-x" else "xy"
    if method.startswith("qaoa-") and config.mixer != mixer:
        raise ValueError("Method and configured mixer disagree")
    if method == "qaoa-xy-untrained" and config.steps != 0:
        raise ValueError("Untrained QAOA requires steps=0")
    return config


def _solve(problem, method, config, seed, deadline, publish, phase):
    """Worker-side dispatch; all method-specific preparation is inside timing."""
    mu, sigma, lam, k = problem.mu, problem.Sigma, problem.lam, problem.k
    if method == "binary-vqe":
        from .binary import run_binary_vqe

        run_binary_vqe(
            mu, sigma, config, candidate_callback=publish, phase_callback=phase
        )
    elif method == "f-vqe":
        from .fvqe import run_fvqe

        run_fvqe(mu, sigma, config, candidate_callback=publish, phase_callback=phase)
    elif method == "qaoa-xy-untrained":
        from .qaoa import _sample_initial_qaoa

        return _sample_initial_qaoa(mu, sigma, config, deadline, publish, phase)
    elif method.startswith("qaoa-"):
        from .qaoa import _sample_qaoa_parameters, run_qaoa

        result = run_qaoa(
            mu, sigma, config, candidate_callback=publish, phase_callback=phase
        )
        if method == "qaoa-xy-trained":
            phase("setup")
            return _sample_qaoa_parameters(
                mu, sigma, config, result.params, deadline, publish, phase
            )
    elif method in {"fractional-vqe", "variational-allocator"}:
        from .fractional import run_fractional_vqe

        run_fractional_vqe(
            mu, sigma, config, candidate_callback=publish, phase_callback=phase
        )
    elif method == "random-k":
        rng = np.random.default_rng(seed)
        phase("candidate_processing")
        seen = set()
        while time.perf_counter() < deadline:
            candidate = np.zeros(len(mu), dtype=int)
            candidate[rng.choice(len(mu), k, replace=False)] = 1
            seen.add(tuple(candidate))
            publish(
                candidate,
                {"source": "uniform_exact_k", "unique_feasible_samples": len(seen)},
            )
            if len(seen) == comb(len(mu), k):
                return "exhausted_feasible_space"
    else:
        from .baselines.solvers import (
            convex_markowitz_baseline,
            frank_wolfe_baseline,
            grasp_binary_baseline,
            iterated_local_search_binary_baseline,
            local_search_binary_baseline,
            memetic_genetic_binary_baseline,
            mixed_integer_binary_baseline,
            parallel_tempering_binary_baseline,
            projected_gradient_descent_baseline,
            relaxation_rounding_binary_baseline,
            scip_binary_baseline,
            simulated_annealing_binary_baseline,
            tabu_search_binary_baseline,
            variable_neighborhood_search_binary_baseline,
        )

        phase("optimization")
        if method in {"highs", "scip"}:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return "time_limit"
            if method == "scip":
                result = scip_binary_baseline(
                    mu, sigma, lam, k, time_limit=remaining, candidate_callback=publish
                )
            else:
                result = mixed_integer_binary_baseline(
                    mu, sigma, lam, k, time_limit=remaining
                )
        elif method == "osqp":
            result = convex_markowitz_baseline(mu, sigma, lam)
        elif method == "pgd":
            result = projected_gradient_descent_baseline(
                mu, sigma, lam, candidate_callback=publish
            )
        elif method == "frank-wolfe":
            result = frank_wolfe_baseline(mu, sigma, lam, candidate_callback=publish)
        elif method == "qp-round":
            result = relaxation_rounding_binary_baseline(
                mu, sigma, lam, k, candidate_callback=publish
            )
        elif method == "swap":
            result = local_search_binary_baseline(
                mu, sigma, lam, k, seed=seed, candidate_callback=publish
            )
        else:
            solve = {
                "anneal": simulated_annealing_binary_baseline,
                "parallel-tempering": parallel_tempering_binary_baseline,
                "tabu": tabu_search_binary_baseline,
                "memetic": memetic_genetic_binary_baseline,
                "ils": iterated_local_search_binary_baseline,
                "grasp": grasp_binary_baseline,
                "vns": variable_neighborhood_search_binary_baseline,
            }[method]
            result = solve(mu, sigma, lam, k, seed=seed, candidate_callback=publish)
        publish(result.candidate, {"source": method, "solver_status": result.status})
        return (
            (result.status, result.diagnostics) if result.diagnostics else result.status
        )
    return "completed"


def _budget_worker(connection, snapshot, method, config, seed, started, budget):
    active_phase = None

    def phase(name):
        nonlocal active_phase
        if name != active_phase:
            connection.send(("phase", (name, time.perf_counter() - started)))
            active_phase = name

    try:
        phase("setup")
        problem = PortfolioProblem.from_dict(snapshot)
        best = float("inf")

        def publish(candidate, details):
            nonlocal best
            previous_phase = active_phase
            phase("candidate_processing")
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
            phase(previous_phase)

        status = _solve(problem, method, config, seed, started + budget, publish, phase)
        diagnostics = {}
        if isinstance(status, tuple):
            status, diagnostics = status
        phase("finished")
        connection.send(("done", (status, "", diagnostics)))
    except Exception as exc:
        phase("finished")
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
    solver_diagnostics = {}
    timing = _PhaseTimings()
    delivery_times = []
    parent_evaluation_seconds = 0.0
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
                evaluation_started = time.perf_counter()
                evaluation = problem.evaluate(payload.candidate)
                parent_evaluation_seconds += time.perf_counter() - evaluation_started
                # Parent validation is part of producing a usable candidate.
                validated = time.perf_counter() - started
                if validated > budget_seconds:
                    late += 1
                    continue
                if not evaluation.feasible:
                    invalid += 1
                elif not incumbents or evaluation.objective < incumbents[-1].objective:
                    incumbents.append(
                        CandidateSnapshot(
                            validated,
                            payload.candidate,
                            evaluation.objective,
                            payload.details,
                        )
                    )
                    delivery_times.append(max(0.0, received - payload.elapsed_seconds))
            elif kind == "phase":
                phase_name, phase_elapsed = payload
                if received <= budget_seconds and phase_elapsed <= received:
                    timing.mark(phase_name, phase_elapsed)
            elif kind == "invalid":
                invalid += 1
            elif kind == "done":
                if received <= budget_seconds:
                    solver_status, error = payload[:2]
                    solver_diagnostics = payload[2] if len(payload) > 2 else {}
                    status = "error" if error else "completed"
                break
    except Exception as exc:
        status, solver_status, error = (
            "error",
            "execution_error",
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        timing_end = min(time.perf_counter() - started, budget_seconds)
        timings = timing.finish(timing_end)
        timings["interrupted_phase"] = timing.active if status == "time_limit" else ""
        cleanup_started = time.perf_counter()
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
        timings.update(
            cleanup_seconds=time.perf_counter() - cleanup_started,
            parent_evaluation_seconds=parent_evaluation_seconds,
            mean_delivery_seconds=(
                sum(delivery_times) / len(delivery_times) if delivery_times else None
            ),
            delivery_count=len(delivery_times),
        )
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
        timings,
        solver_diagnostics=solver_diagnostics,
    )
    worker.close()
    return result
