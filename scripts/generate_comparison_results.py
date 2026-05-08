#!/usr/bin/env python
"""Generate compact comparison CSVs for synthetic portfolio benchmarks.

The script intentionally uses deterministic synthetic data and NumPy-only
classical baselines so it can run without notebook execution, market-data
downloads, or optional solver dependencies.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from vqe_portfolio import (
    BinaryVQEConfig,
    FractionalVQEConfig,
    QAOAConfig,
    run_binary_vqe,
    run_fractional_vqe,
    run_qaoa,
)
from vqe_portfolio.metrics import portfolio_return, portfolio_risk

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
]


def synthetic_problem(n_assets: int, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + n_assets)
    mu = np.linspace(0.07, 0.18, n_assets) + rng.normal(0.0, 0.008, n_assets)
    factors = rng.normal(0.0, 0.08, size=(n_assets, min(3, n_assets)))
    Sigma = factors @ factors.T + np.diag(np.linspace(0.03, 0.09, n_assets))
    return mu.astype(float), Sigma.astype(float)


def binary_objective(
    x: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    alpha: float,
    k: int,
) -> float:
    x = np.asarray(x, dtype=float)
    return float(lam * x @ Sigma @ x - mu @ x + alpha * (x.sum() - k) ** 2)


def fractional_objective(
    w: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
) -> float:
    w = np.asarray(w, dtype=float)
    return float(-mu @ w + lam * w @ Sigma @ w)


def exhaustive_binary_baseline(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    alpha: float,
    k: int,
) -> tuple[np.ndarray, float]:
    best_x: np.ndarray | None = None
    best_obj = float("inf")
    for combo in itertools.combinations(range(len(mu)), k):
        x = np.zeros(len(mu), dtype=int)
        x[list(combo)] = 1
        obj = binary_objective(x, mu, Sigma, lam=lam, alpha=alpha, k=k)
        if obj < best_obj:
            best_x = x
            best_obj = obj
    if best_x is None:
        raise RuntimeError("No feasible binary portfolio was generated.")
    return best_x, best_obj


def top_return_baseline(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    alpha: float,
    k: int,
) -> tuple[np.ndarray, float]:
    selected = np.argsort(mu)[-k:]
    x = np.zeros(len(mu), dtype=int)
    x[selected] = 1
    return x, binary_objective(x, mu, Sigma, lam=lam, alpha=alpha, k=k)


def minimum_variance_binary_baseline(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    alpha: float,
    k: int,
) -> tuple[np.ndarray, float]:
    best_x: np.ndarray | None = None
    best_variance = float("inf")
    for combo in itertools.combinations(range(len(mu)), k):
        x = np.zeros(len(mu), dtype=int)
        x[list(combo)] = 1
        variance = float(x @ Sigma @ x)
        if variance < best_variance:
            best_x = x
            best_variance = variance
    if best_x is None:
        raise RuntimeError("No feasible binary portfolio was generated.")
    return best_x, binary_objective(best_x, mu, Sigma, lam=lam, alpha=alpha, k=k)


def exact_long_only_markowitz_baseline(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    tol: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """Solve the long-only simplex Markowitz problem by active-set enumeration."""
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)

    if lam <= 0:
        w = np.zeros(n, dtype=float)
        w[int(np.argmax(mu))] = 1.0
        return w, fractional_objective(w, mu, Sigma, lam=lam)

    best_w: np.ndarray | None = None
    best_obj = float("inf")

    for size in range(1, n + 1):
        for support in itertools.combinations(range(n), size):
            idx = np.array(support, dtype=int)
            sigma_s = Sigma[np.ix_(idx, idx)]
            mu_s = mu[idx]
            lhs = np.block(
                [
                    [2.0 * lam * sigma_s, np.ones((size, 1))],
                    [np.ones((1, size)), np.zeros((1, 1))],
                ]
            )
            rhs = np.concatenate([mu_s, np.array([1.0])])
            try:
                sol = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            w_s = sol[:size]
            if np.any(w_s < -tol):
                continue
            w_s = np.maximum(w_s, 0.0)
            total = float(w_s.sum())
            if total <= 0:
                continue
            w_s = w_s / total

            w = np.zeros(n, dtype=float)
            w[idx] = w_s
            obj = fractional_objective(w, mu, Sigma, lam=lam)
            if obj < best_obj:
                best_w = w
                best_obj = obj

    if best_w is None:
        raise RuntimeError("No feasible Markowitz allocation was generated.")

    return best_w, best_obj


def equal_weight_baseline(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, float]:
    w = np.full(len(mu), 1.0 / len(mu), dtype=float)
    return w, fractional_objective(w, mu, Sigma, lam=lam)


def bitstring(x: np.ndarray) -> str:
    return "".join(str(int(v)) for v in np.asarray(x, dtype=int))


def weights_string(w: np.ndarray) -> str:
    return "[" + " ".join(f"{float(v):.6f}" for v in np.asarray(w, dtype=float)) + "]"


def feasible_sample_rate(sample_counts: dict[str, int], k: int) -> float:
    total = sum(sample_counts.values())
    if total == 0:
        return 0.0
    feasible = sum(
        count for bits, count in sample_counts.items() if bits.count("1") == k
    )
    return feasible / total


def selected_equal_weight(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    selected = float(x.sum())
    if selected <= 0:
        return x
    return x / selected


def summarize_trials(
    *,
    dataset: str,
    n_assets: int,
    k: int | None,
    method: str,
    method_type: str,
    lam: float,
    objective_family: str,
    reported_weighting: str,
    objectives: list[float],
    returns: list[float],
    risks: list[float],
    feasible_rates: list[float],
    runtimes: list[float],
    candidates: list[str],
    notes: str,
) -> dict[str, str]:
    best_idx = int(np.argmin(objectives))
    return {
        "dataset": dataset,
        "n_assets": str(n_assets),
        "k": "" if k is None else str(k),
        "method": method,
        "type": method_type,
        "lambda": f"{lam:.6g}",
        "objective_family": objective_family,
        "reported_weighting": reported_weighting,
        "seed_count": str(len(objectives)),
        "best_objective": f"{min(objectives):.6f}",
        "mean_objective": f"{mean(objectives):.6f}",
        "std_objective": (
            f"{pstdev(objectives):.6f}" if len(objectives) > 1 else "0.000000"
        ),
        "mean_return": f"{mean(returns):.6f}",
        "mean_risk": f"{mean(risks):.6f}",
        "feasible_rate": f"{mean(feasible_rates):.6f}",
        "mean_runtime_seconds": f"{mean(runtimes):.6f}",
        "best_selection_or_weights": candidates[best_idx],
        "notes": notes,
    }


def trial_row(
    *,
    dataset: str,
    n_assets: int,
    k: int | None,
    method: str,
    method_type: str,
    lam: float,
    objective_family: str,
    reported_weighting: str,
    seed: int | str,
    objective: float,
    ret: float,
    risk: float,
    feasible_rate: float,
    runtime: float,
    candidate: str,
    notes: str,
) -> dict[str, str]:
    return {
        "dataset": dataset,
        "n_assets": str(n_assets),
        "k": "" if k is None else str(k),
        "method": method,
        "type": method_type,
        "lambda": f"{lam:.6g}",
        "objective_family": objective_family,
        "reported_weighting": reported_weighting,
        "seed": str(seed),
        "objective": f"{objective:.6f}",
        "return": f"{ret:.6f}",
        "risk": f"{risk:.6f}",
        "feasible_rate": f"{feasible_rate:.6f}",
        "runtime_seconds": f"{runtime:.6f}",
        "selection_or_weights": candidate,
        "notes": notes,
    }


def add_summary_and_trial(
    rows: list[dict[str, str]],
    trial_rows: list[dict[str, str]],
    *,
    dataset: str,
    n_assets: int,
    k: int | None,
    method: str,
    method_type: str,
    lam: float,
    objective_family: str,
    reported_weighting: str,
    objective: float,
    ret: float,
    risk: float,
    feasible_rate: float,
    runtime: float,
    candidate: str,
    notes: str,
) -> None:
    rows.append(
        summarize_trials(
            dataset=dataset,
            n_assets=n_assets,
            k=k,
            method=method,
            method_type=method_type,
            lam=lam,
            objective_family=objective_family,
            reported_weighting=reported_weighting,
            objectives=[objective],
            returns=[ret],
            risks=[risk],
            feasible_rates=[feasible_rate],
            runtimes=[runtime],
            candidates=[candidate],
            notes=notes,
        )
    )
    trial_rows.append(
        trial_row(
            dataset=dataset,
            n_assets=n_assets,
            k=k,
            method=method,
            method_type=method_type,
            lam=lam,
            objective_family=objective_family,
            reported_weighting=reported_weighting,
            seed="-",
            objective=objective,
            ret=ret,
            risk=risk,
            feasible_rate=feasible_rate,
            runtime=runtime,
            candidate=candidate,
            notes=notes,
        )
    )


def add_classical_rows(
    rows: list[dict[str, str]],
    trial_rows: list[dict[str, str]],
    dataset: str,
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    alpha: float,
    k: int,
) -> None:
    started = time.perf_counter()
    x, obj = exhaustive_binary_baseline(mu, Sigma, lam=lam, alpha=alpha, k=k)
    elapsed = time.perf_counter() - started
    add_summary_and_trial(
        rows,
        trial_rows,
        dataset=dataset,
        n_assets=len(mu),
        k=k,
        method="Classical exhaustive search",
        method_type="classical",
        lam=lam,
        objective=obj,
        objective_family="binary_qubo",
        reported_weighting="equal_weight_selected",
        ret=portfolio_return(mu, selected_equal_weight(x)),
        risk=portfolio_risk(Sigma, selected_equal_weight(x)),
        feasible_rate=1.0,
        runtime=elapsed,
        candidate=bitstring(x),
        notes="Exact binary cardinality baseline",
    )

    for method, baseline, notes in [
        (
            "Classical top-return heuristic",
            top_return_baseline,
            "Selects the K largest expected returns",
        ),
        (
            "Classical minimum-variance subset",
            minimum_variance_binary_baseline,
            "Selects the feasible subset with the lowest quadratic variance",
        ),
    ]:
        started = time.perf_counter()
        x, obj = baseline(mu, Sigma, lam=lam, alpha=alpha, k=k)
        elapsed = time.perf_counter() - started
        add_summary_and_trial(
            rows,
            trial_rows,
            dataset=dataset,
            n_assets=len(mu),
            k=k,
            method=method,
            method_type="classical heuristic",
            lam=lam,
            objective=obj,
            objective_family="binary_qubo",
            reported_weighting="equal_weight_selected",
            ret=portfolio_return(mu, selected_equal_weight(x)),
            risk=portfolio_risk(Sigma, selected_equal_weight(x)),
            feasible_rate=1.0,
            runtime=elapsed,
            candidate=bitstring(x),
            notes=notes,
        )

    started = time.perf_counter()
    w, obj = equal_weight_baseline(mu, Sigma, lam=lam)
    elapsed = time.perf_counter() - started
    add_summary_and_trial(
        rows,
        trial_rows,
        dataset=dataset,
        n_assets=len(mu),
        k=None,
        method="Classical equal weight",
        method_type="classical heuristic",
        lam=lam,
        objective_family="fractional_simplex",
        reported_weighting="simplex_weights",
        objective=obj,
        ret=portfolio_return(mu, w),
        risk=portfolio_risk(Sigma, w),
        feasible_rate=1.0,
        runtime=elapsed,
        candidate=weights_string(w),
        notes="Uniform long-only allocation baseline",
    )

    started = time.perf_counter()
    w, obj = exact_long_only_markowitz_baseline(mu, Sigma, lam=lam)
    elapsed = time.perf_counter() - started
    add_summary_and_trial(
        rows,
        trial_rows,
        dataset=dataset,
        n_assets=len(mu),
        k=None,
        method="Classical exact Markowitz",
        method_type="classical",
        lam=lam,
        objective_family="fractional_simplex",
        reported_weighting="simplex_weights",
        objective=obj,
        ret=portfolio_return(mu, w),
        risk=portfolio_risk(Sigma, w),
        feasible_rate=1.0,
        runtime=elapsed,
        candidate=weights_string(w),
        notes="Exact long-only simplex baseline from active-set enumeration",
    )


def add_binary_vqe_row(
    rows: list[dict[str, str]],
    trial_rows: list[dict[str, str]],
    dataset: str,
    mu: np.ndarray,
    Sigma: np.ndarray,
    base_cfg: BinaryVQEConfig,
    seeds: list[int],
) -> None:
    objectives: list[float] = []
    returns: list[float] = []
    risks: list[float] = []
    feasible_rates: list[float] = []
    runtimes: list[float] = []
    candidates: list[str] = []
    method = "Binary VQE best feasible"
    notes = "Falls back to Top-K projection if no feasible sample is observed"

    for seed in seeds:
        started = time.perf_counter()
        res = run_binary_vqe(mu, Sigma, replace(base_cfg, seed=seed))
        elapsed = time.perf_counter() - started
        x = np.asarray(
            res.x_best_feasible if res.x_best_feasible is not None else res.x_topk
        )
        w = selected_equal_weight(x)
        objectives.append(
            binary_objective(x, mu, Sigma, base_cfg.lam, base_cfg.alpha, base_cfg.k)
        )
        returns.append(portfolio_return(mu, w))
        risks.append(portfolio_risk(Sigma, w))
        feasible_rates.append(feasible_sample_rate(res.sample_counts, base_cfg.k))
        runtimes.append(elapsed)
        candidates.append(bitstring(x))
        trial_rows.append(
            trial_row(
                dataset=dataset,
                n_assets=len(mu),
                k=base_cfg.k,
                method=method,
                method_type="quantum",
                lam=base_cfg.lam,
                objective_family="binary_qubo",
                reported_weighting="equal_weight_selected",
                seed=seed,
                objective=objectives[-1],
                ret=returns[-1],
                risk=risks[-1],
                feasible_rate=feasible_rates[-1],
                runtime=elapsed,
                candidate=candidates[-1],
                notes=notes,
            )
        )

    rows.append(
        summarize_trials(
            dataset=dataset,
            n_assets=len(mu),
            k=base_cfg.k,
            method=method,
            method_type="quantum",
            lam=base_cfg.lam,
            objective_family="binary_qubo",
            reported_weighting="equal_weight_selected",
            objectives=objectives,
            returns=returns,
            risks=risks,
            feasible_rates=feasible_rates,
            runtimes=runtimes,
            candidates=candidates,
            notes=notes,
        )
    )


def add_qaoa_row(
    rows: list[dict[str, str]],
    trial_rows: list[dict[str, str]],
    dataset: str,
    mu: np.ndarray,
    Sigma: np.ndarray,
    base_cfg: QAOAConfig,
    seeds: list[int],
) -> None:
    objectives: list[float] = []
    returns: list[float] = []
    risks: list[float] = []
    feasible_rates: list[float] = []
    runtimes: list[float] = []
    candidates: list[str] = []
    method = f"QAOA {base_cfg.mixer.upper()} best feasible"
    notes = "Falls back to Top-K projection if no feasible sample is observed"

    for seed in seeds:
        started = time.perf_counter()
        res = run_qaoa(mu, Sigma, replace(base_cfg, seed=seed))
        elapsed = time.perf_counter() - started
        x = np.asarray(
            res.x_best_feasible if res.x_best_feasible is not None else res.x_topk
        )
        w = selected_equal_weight(x)
        objectives.append(
            binary_objective(x, mu, Sigma, base_cfg.lam, base_cfg.alpha, base_cfg.k)
        )
        returns.append(portfolio_return(mu, w))
        risks.append(portfolio_risk(Sigma, w))
        feasible_rates.append(feasible_sample_rate(res.sample_counts, base_cfg.k))
        runtimes.append(elapsed)
        candidates.append(bitstring(x))
        trial_rows.append(
            trial_row(
                dataset=dataset,
                n_assets=len(mu),
                k=base_cfg.k,
                method=method,
                method_type="quantum",
                lam=base_cfg.lam,
                objective_family="binary_qubo",
                reported_weighting="equal_weight_selected",
                seed=seed,
                objective=objectives[-1],
                ret=returns[-1],
                risk=risks[-1],
                feasible_rate=feasible_rates[-1],
                runtime=elapsed,
                candidate=candidates[-1],
                notes=notes,
            )
        )

    rows.append(
        summarize_trials(
            dataset=dataset,
            n_assets=len(mu),
            k=base_cfg.k,
            method=method,
            method_type="quantum",
            lam=base_cfg.lam,
            objective_family="binary_qubo",
            reported_weighting="equal_weight_selected",
            objectives=objectives,
            returns=returns,
            risks=risks,
            feasible_rates=feasible_rates,
            runtimes=runtimes,
            candidates=candidates,
            notes=notes,
        )
    )


def add_fractional_vqe_row(
    rows: list[dict[str, str]],
    trial_rows: list[dict[str, str]],
    dataset: str,
    mu: np.ndarray,
    Sigma: np.ndarray,
    base_cfg: FractionalVQEConfig,
    seeds: list[int],
) -> None:
    objectives: list[float] = []
    returns: list[float] = []
    risks: list[float] = []
    feasible_rates: list[float] = []
    runtimes: list[float] = []
    candidates: list[str] = []
    method = "Fractional VQE"
    notes = "Simplex-normalized allocation"

    for seed in seeds:
        started = time.perf_counter()
        res = run_fractional_vqe(mu, Sigma, replace(base_cfg, seed=seed))
        elapsed = time.perf_counter() - started
        w = np.asarray(res.weights, dtype=float)
        objectives.append(fractional_objective(w, mu, Sigma, base_cfg.lam))
        returns.append(portfolio_return(mu, w))
        risks.append(portfolio_risk(Sigma, w))
        feasible_rates.append(float(np.all(w >= -1e-8) and abs(w.sum() - 1.0) < 1e-6))
        runtimes.append(elapsed)
        candidates.append(weights_string(w))
        trial_rows.append(
            trial_row(
                dataset=dataset,
                n_assets=len(mu),
                k=None,
                method=method,
                method_type="quantum",
                lam=base_cfg.lam,
                objective_family="fractional_simplex",
                reported_weighting="simplex_weights",
                seed=seed,
                objective=objectives[-1],
                ret=returns[-1],
                risk=risks[-1],
                feasible_rate=feasible_rates[-1],
                runtime=elapsed,
                candidate=candidates[-1],
                notes=notes,
            )
        )

    rows.append(
        summarize_trials(
            dataset=dataset,
            n_assets=len(mu),
            k=None,
            method=method,
            method_type="quantum",
            lam=base_cfg.lam,
            objective_family="fractional_simplex",
            reported_weighting="simplex_weights",
            objectives=objectives,
            returns=returns,
            risks=risks,
            feasible_rates=feasible_rates,
            runtimes=runtimes,
            candidates=candidates,
            notes=notes,
        )
    )


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_methods(value: str) -> set[str]:
    methods = {item.strip().lower() for item in value.split(",") if item.strip()}
    if "all" in methods:
        return {"classical", "binary-vqe", "qaoa-x", "qaoa-xy", "fractional-vqe"}
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/generated_comparison_summary.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--trials-output",
        default="results/generated_repeatability_trials.csv",
        help="Per-seed trial CSV output path.",
    )
    parser.add_argument(
        "--asset-counts",
        default="4",
        help="Comma-separated synthetic problem sizes.",
    )
    parser.add_argument(
        "--seeds", default="0,1,2", help="Comma-separated quantum seeds."
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated subset: classical,binary-vqe,qaoa-x,qaoa-xy,fractional-vqe.",
    )
    parser.add_argument("--lambda-value", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--binary-depth", type=int, default=2)
    parser.add_argument("--qaoa-p", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asset_counts = parse_int_list(args.asset_counts)
    seeds = parse_int_list(args.seeds)
    methods = parse_methods(args.methods)

    rows: list[dict[str, str]] = []
    trial_rows: list[dict[str, str]] = []
    for n_assets in asset_counts:
        k = max(1, min(2, n_assets - 1))
        dataset = f"Synthetic generated n={n_assets}"
        mu, Sigma = synthetic_problem(n_assets)

        if "classical" in methods:
            add_classical_rows(
                rows,
                trial_rows,
                dataset,
                mu,
                Sigma,
                args.lambda_value,
                args.alpha,
                k,
            )

        if "binary-vqe" in methods:
            cfg = BinaryVQEConfig(
                depth=args.binary_depth,
                steps=args.steps,
                stepsize=0.3,
                log_every=max(args.steps, 1),
                lam=args.lambda_value,
                alpha=args.alpha,
                k=k,
                shots_train=None,
                shots_sample=args.shots,
            )
            add_binary_vqe_row(rows, trial_rows, dataset, mu, Sigma, cfg, seeds)

        if "qaoa-x" in methods:
            cfg = QAOAConfig(
                p=args.qaoa_p,
                steps=args.steps,
                stepsize=0.2,
                log_every=max(args.steps, 1),
                lam=args.lambda_value,
                alpha=args.alpha,
                k=k,
                mixer="x",
                shots_train=None,
                shots_sample=args.shots,
            )
            add_qaoa_row(rows, trial_rows, dataset, mu, Sigma, cfg, seeds)

        if "qaoa-xy" in methods:
            cfg = QAOAConfig(
                p=args.qaoa_p,
                steps=args.steps,
                stepsize=0.2,
                log_every=max(args.steps, 1),
                lam=args.lambda_value,
                alpha=args.alpha,
                k=k,
                mixer="xy",
                shots_train=None,
                shots_sample=args.shots,
            )
            add_qaoa_row(rows, trial_rows, dataset, mu, Sigma, cfg, seeds)

        if "fractional-vqe" in methods:
            cfg = FractionalVQEConfig(
                steps=args.steps,
                stepsize=0.3,
                log_every=max(args.steps, 1),
                lam=args.lambda_value,
                shots=None,
            )
            add_fractional_vqe_row(rows, trial_rows, dataset, mu, Sigma, cfg, seeds)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    trials_output = Path(args.trials_output)
    trials_output.parent.mkdir(parents=True, exist_ok=True)
    with trials_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(trial_rows)

    print(f"Wrote {len(rows)} summary rows to {output}")
    print(f"Wrote {len(trial_rows)} trial rows to {trials_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
