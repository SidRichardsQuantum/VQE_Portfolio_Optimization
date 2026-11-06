"""Synthetic and market-data comparison workflows; evaluation lives in benchmark."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Iterable

import numpy as np

from ..baselines import (
    convex_markowitz_baseline,
    frank_wolfe_baseline,
    equal_weight_baseline,
    exact_long_only_markowitz_baseline,
    exhaustive_binary_baseline,
    grasp_binary_baseline,
    iterated_local_search_binary_baseline,
    local_search_binary_baseline,
    memetic_genetic_binary_baseline,
    minimum_variance_binary_baseline,
    mixed_integer_binary_baseline,
    parallel_tempering_binary_baseline,
    projected_gradient_descent_baseline,
    relaxation_rounding_binary_baseline,
    scip_binary_baseline,
    simulated_annealing_binary_baseline,
    tabu_search_binary_baseline,
    variable_neighborhood_search_binary_baseline,
    top_return_baseline,
)
from ..benchmark import (
    CandidateTrial,
    comparison_rows,
    feasible_sample_rate,
    labels_from_selection,
)
from ..binary import run_binary_vqe
from ..data import get_stock_data
from ..fractional import run_fractional_vqe
from ..fvqe import run_fvqe
from ..problem import PortfolioProblem
from ..qaoa import run_qaoa
from ..reliability import resolve_penalty
from ..types import BinaryVQEConfig, FractionalVQEConfig, QAOAConfig

FIELDNAMES = [
    "dataset",
    "n_assets",
    "k",
    "method",
    "type",
    "lambda",
    "objective_family",
    "reported_weighting",
    "seed_count",
    "best_objective",
    "mean_objective",
    "std_objective",
    "mean_return",
    "mean_risk",
    "feasible_rate",
    "mean_runtime_seconds",
    "best_selection_or_weights",
    "notes",
    "solver_status",
    "lower_bound",
    "mip_gap",
    "problem_id",
    "sample_feasible_rate",
]

TRIAL_FIELDNAMES = [
    "dataset",
    "n_assets",
    "k",
    "method",
    "type",
    "lambda",
    "objective_family",
    "reported_weighting",
    "seed",
    "objective",
    "return",
    "risk",
    "feasible_rate",
    "runtime_seconds",
    "selection_or_weights",
    "notes",
    "solver_status",
    "lower_bound",
    "mip_gap",
    "problem_id",
    "sample_feasible_rate",
]

TRIAL_FIELDNAMES += ["penalized_energy", "validation_error"]


def synthetic_problem(n_assets: int, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + n_assets)
    mu = np.linspace(0.07, 0.18, n_assets) + rng.normal(0.0, 0.008, n_assets)
    factors = rng.normal(0.0, 0.08, size=(n_assets, min(3, n_assets)))
    Sigma = factors @ factors.T + np.diag(np.linspace(0.03, 0.09, n_assets))
    return mu.astype(float), Sigma.astype(float)


def _append(
    rows, trial_rows, problem, trials, dataset, method, method_type, alpha=None
):
    summary, runs = comparison_rows(
        problem,
        trials,
        dataset=dataset,
        method=method,
        method_type=method_type,
        alpha=alpha,
    )
    rows.append(summary)
    trial_rows.extend(runs)


def add_classical_rows(rows, trial_rows, dataset, mu, Sigma, lam, alpha, k):
    binary = PortfolioProblem(mu, Sigma, lam, "binary_qubo", k)
    fractional = PortfolioProblem(mu, Sigma, lam, "fractional_simplex")
    for method, baseline, problem, method_type in [
        (
            "Classical exhaustive search",
            exhaustive_binary_baseline,
            binary,
            "classical",
        ),
        (
            "Classical top-return heuristic",
            top_return_baseline,
            binary,
            "classical heuristic",
        ),
        (
            "Classical minimum-variance subset",
            minimum_variance_binary_baseline,
            binary,
            "classical heuristic",
        ),
        (
            "Classical equal weight",
            equal_weight_baseline,
            fractional,
            "classical heuristic",
        ),
        (
            "Classical exact Markowitz",
            exact_long_only_markowitz_baseline,
            fractional,
            "classical",
        ),
    ]:
        kwargs = {"alpha": alpha, "k": k} if problem.k is not None else {}
        started = time.perf_counter()
        candidate, _ = baseline(problem.mu, problem.Sigma, lam=problem.lam, **kwargs)
        trial = CandidateTrial(candidate, time.perf_counter() - started)
        _append(
            rows,
            trial_rows,
            problem,
            [trial],
            dataset,
            method,
            method_type,
            alpha if problem.k is not None else None,
        )


def add_solver_rows(
    rows,
    trial_rows,
    dataset,
    mu,
    Sigma,
    lam,
    k,
    seeds,
    time_limit=60.0,
    restarts=16,
    *,
    methods=None,
):
    """Add all strong baselines or select their budget-method IDs explicitly."""
    binary = PortfolioProblem(mu, Sigma, lam, "binary_qubo", k)
    fractional = PortfolioProblem(mu, Sigma, lam, "fractional_simplex")
    for method_id, method, problem, run_seeds, solve in [
        (
            "osqp",
            "Classical convex Markowitz (OSQP)",
            fractional,
            ["-"],
            lambda p, seed: convex_markowitz_baseline(p.mu, p.Sigma, p.lam),
        ),
        (
            "highs",
            "Classical mixed integer (HiGHS)",
            binary,
            ["-"],
            lambda p, seed: mixed_integer_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k, time_limit=time_limit
            ),
        ),
        (
            "scip",
            "Classical quadratic integer (SCIP)",
            binary,
            ["-"],
            lambda p, seed: scip_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k, time_limit=time_limit
            ),
        ),
        (
            "qp-round",
            "Classical QP relaxation + rounding",
            binary,
            ["-"],
            lambda p, seed: relaxation_rounding_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k
            ),
        ),
        (
            "swap",
            "Classical multistart swap search",
            binary,
            seeds,
            lambda p, seed: local_search_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k, seed=seed, restarts=restarts
            ),
        ),
        (
            "anneal",
            "Classical simulated annealing",
            binary,
            seeds,
            lambda p, seed: simulated_annealing_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k, seed=seed, restarts=restarts
            ),
        ),
        (
            "tabu",
            "Classical tabu search",
            binary,
            seeds,
            lambda p, seed: tabu_search_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k, seed=seed, restarts=restarts
            ),
        ),
        (
            "memetic",
            "Classical memetic genetic algorithm",
            binary,
            seeds,
            lambda p, seed: memetic_genetic_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k, seed=seed, restarts=restarts
            ),
        ),
        (
            "ils",
            "Classical iterated local search",
            binary,
            seeds,
            lambda p, seed: iterated_local_search_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k, seed=seed, restarts=restarts
            ),
        ),
        (
            "grasp",
            "Classical GRASP",
            binary,
            seeds,
            lambda p, seed: grasp_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k, seed=seed, restarts=restarts
            ),
        ),
        (
            "vns",
            "Classical variable neighborhood search",
            binary,
            seeds,
            lambda p, seed: variable_neighborhood_search_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k, seed=seed, restarts=restarts
            ),
        ),
        (
            "pgd",
            "Classical projected gradient descent",
            fractional,
            ["-"],
            lambda p, seed: projected_gradient_descent_baseline(p.mu, p.Sigma, p.lam),
        ),
        (
            "frank-wolfe",
            "Classical Frank-Wolfe with away steps",
            fractional,
            ["-"],
            lambda p, seed: frank_wolfe_baseline(p.mu, p.Sigma, p.lam),
        ),
        (
            "parallel-tempering",
            "Classical parallel tempering",
            binary,
            seeds,
            lambda p, seed: parallel_tempering_binary_baseline(
                p.mu, p.Sigma, p.lam, p.k, seed=seed, restarts=restarts
            ),
        ),
    ]:
        if methods is not None and method_id not in methods:
            continue
        trials = []
        for seed in run_seeds:
            started = time.perf_counter()
            result = solve(problem, seed)
            trials.append(
                CandidateTrial(
                    result.candidate,
                    time.perf_counter() - started,
                    seed=seed,
                    status=result.status,
                    lower_bound=result.lower_bound,
                    mip_gap=result.mip_gap,
                    notes=result.message,
                )
            )
        _append(
            rows,
            trial_rows,
            problem,
            trials,
            dataset,
            method,
            (
                "classical heuristic"
                if method_id
                in {
                    "qp-round",
                    "swap",
                    "anneal",
                    "parallel-tempering",
                    "tabu",
                    "memetic",
                    "ils",
                    "grasp",
                    "vns",
                }
                else "classical"
            ),
        )


def _add_quantum_row(
    rows, trial_rows, dataset, mu, Sigma, cfg, seeds, solve, method, binary
):
    problem = PortfolioProblem(
        mu,
        Sigma,
        cfg.lam,
        "binary_qubo" if binary else "fractional_simplex",
        cfg.k if binary else None,
    )
    trials = []
    for seed in seeds:
        started = time.perf_counter()
        result = solve(problem.mu, problem.Sigma, replace(cfg, seed=seed))
        if binary:
            fallback = result.x_best_feasible is None
            candidate = result.x_topk if fallback else result.x_best_feasible
            sample_rate = feasible_sample_rate(result.sample_counts, problem.k)
            notes = "Top-K fallback" if fallback else "Best feasible sampled candidate"
        else:
            candidate, sample_rate, notes = (
                result.weights,
                None,
                "Returned simplex allocation",
            )
        trials.append(
            CandidateTrial(
                candidate,
                time.perf_counter() - started,
                seed=seed,
                sample_feasible_rate=sample_rate,
                notes=notes,
            )
        )
    _append(
        rows,
        trial_rows,
        problem,
        trials,
        dataset,
        method,
        "quantum",
        resolve_penalty(mu, Sigma, cfg.lam, cfg.k, cfg.alpha) if binary else None,
    )


def add_binary_vqe_row(rows, trial_rows, dataset, mu, Sigma, base_cfg, seeds):
    _add_quantum_row(
        rows,
        trial_rows,
        dataset,
        mu,
        Sigma,
        base_cfg,
        seeds,
        run_binary_vqe,
        "Binary VQE best feasible",
        True,
    )


def add_qaoa_row(rows, trial_rows, dataset, mu, Sigma, base_cfg, seeds):
    _add_quantum_row(
        rows,
        trial_rows,
        dataset,
        mu,
        Sigma,
        base_cfg,
        seeds,
        run_qaoa,
        f"QAOA {base_cfg.mixer.upper()} best feasible",
        True,
    )


def add_fvqe_row(rows, trial_rows, dataset, mu, Sigma, base_cfg, seeds):
    _add_quantum_row(
        rows,
        trial_rows,
        dataset,
        mu,
        Sigma,
        base_cfg,
        seeds,
        run_fvqe,
        f"F-VQE {base_cfg.filter_type} best feasible",
        True,
    )


def add_fractional_vqe_row(rows, trial_rows, dataset, mu, Sigma, base_cfg, seeds):
    _add_quantum_row(
        rows,
        trial_rows,
        dataset,
        mu,
        Sigma,
        base_cfg,
        seeds,
        run_fractional_vqe,
        "Variational quantum allocator",
        False,
    )


def real_data_comparison_rows(
    tickers: Iterable[str],
    *,
    start: str,
    end: str,
    lam: float,
    alpha: float,
    k: int,
    seed: int = 0,
    steps: int = 25,
    shots: int = 512,
    shrink: str | None = "lw",
    scale: str = "none",
) -> tuple[list[dict[str, str]], object]:
    tickers = [ticker.upper() for ticker in tickers]
    mu_series, sigma_df, prices = get_stock_data(
        tickers,
        start=start,
        end=end,
        use_log=True,
        shrink=shrink,
        scale=scale,
        progress=False,
    )
    binary_problem = PortfolioProblem(
        mu_series.values, sigma_df.values, lam, "binary_qubo", k
    )
    fractional_problem = PortfolioProblem(
        mu_series.values, sigma_df.values, lam, "fractional_simplex"
    )
    mu, Sigma = binary_problem.mu, binary_problem.Sigma

    def report(
        problem, method, method_type, candidate, runtime, notes, sample_rate=None
    ):
        _, runs = comparison_rows(
            problem,
            [
                CandidateTrial(
                    candidate,
                    runtime,
                    seed=seed,
                    sample_feasible_rate=sample_rate,
                    notes=notes,
                )
            ],
            dataset="Real data method comparison",
            method=method,
            method_type=method_type,
            alpha=alpha if problem.k is not None else None,
        )
        row = runs[0]
        row["window"] = f"{start} to {end}"
        row["bitstring"] = ""
        if problem.k is not None and row["feasible_rate"] == "1.000000":
            row["bitstring"] = row["selection_or_weights"]
            row["selection_or_weights"] = labels_from_selection(tickers, candidate)
        return row

    def binary_row(method, method_type, x, runtime, feasible_rate, notes):
        return report(
            binary_problem,
            method,
            method_type,
            x,
            runtime,
            notes,
            feasible_rate if method_type == "quantum" else None,
        )

    def weight_row(method, method_type, w, runtime, notes):
        return report(fractional_problem, method, method_type, w, runtime, notes)

    rows: list[dict[str, str]] = []

    started = time.perf_counter()
    x_exact, _ = exhaustive_binary_baseline(mu, Sigma, lam=lam, alpha=alpha, k=k)
    rows.append(
        binary_row(
            "Classical exhaustive search",
            "classical",
            x_exact,
            time.perf_counter() - started,
            1.0,
            "Exact cardinality baseline",
        )
    )

    started = time.perf_counter()
    w_equal, _ = equal_weight_baseline(mu, Sigma, lam=lam)
    rows.append(
        weight_row(
            "Classical equal weight",
            "classical heuristic",
            w_equal,
            time.perf_counter() - started,
            "Uniform long-only allocation",
        )
    )

    started = time.perf_counter()
    w_markowitz, _ = exact_long_only_markowitz_baseline(mu, Sigma, lam=lam)
    rows.append(
        weight_row(
            "Classical exact Markowitz",
            "classical",
            w_markowitz,
            time.perf_counter() - started,
            "Exact long-only simplex baseline from active-set enumeration",
        )
    )

    binary_cfg = BinaryVQEConfig(
        depth=2,
        steps=steps,
        stepsize=0.3,
        log_every=max(steps, 1),
        lam=lam,
        alpha=alpha,
        k=k,
        seed=seed,
        shots_train=None,
        shots_sample=shots,
    )
    started = time.perf_counter()
    binary_res = run_binary_vqe(mu, Sigma, binary_cfg)
    x_binary = (
        binary_res.x_best_feasible
        if binary_res.x_best_feasible is not None
        else binary_res.x_topk
    )
    rows.append(
        binary_row(
            "Binary VQE best feasible",
            "quantum",
            x_binary,
            time.perf_counter() - started,
            feasible_sample_rate(binary_res.sample_counts, k),
            (
                "Top-K fallback"
                if binary_res.x_best_feasible is None
                else "Best feasible sampled candidate"
            ),
        )
    )

    for mixer in ["x", "xy"]:
        qaoa_cfg = QAOAConfig(
            p=1,
            steps=steps,
            stepsize=0.2,
            log_every=max(steps, 1),
            lam=lam,
            alpha=alpha,
            k=k,
            mixer=mixer,
            seed=seed,
            shots_train=None,
            shots_sample=shots,
        )
        started = time.perf_counter()
        qaoa_res = run_qaoa(mu, Sigma, qaoa_cfg)
        x_qaoa = (
            qaoa_res.x_best_feasible
            if qaoa_res.x_best_feasible is not None
            else qaoa_res.x_topk
        )
        rows.append(
            binary_row(
                f"QAOA {mixer.upper()} best feasible",
                "quantum",
                x_qaoa,
                time.perf_counter() - started,
                feasible_sample_rate(qaoa_res.sample_counts, k),
                (
                    "Top-K fallback"
                    if qaoa_res.x_best_feasible is None
                    else "Best feasible sampled candidate"
                ),
            )
        )

    fractional_cfg = FractionalVQEConfig(
        steps=steps,
        stepsize=0.3,
        log_every=max(steps, 1),
        lam=lam,
        seed=seed,
        shots=None,
    )
    started = time.perf_counter()
    fractional_res = run_fractional_vqe(mu, Sigma, fractional_cfg)
    rows.append(
        weight_row(
            "Variational quantum allocator",
            "quantum",
            fractional_res.weights,
            time.perf_counter() - started,
            "Simplex-normalized allocation",
        )
    )

    return rows, prices
