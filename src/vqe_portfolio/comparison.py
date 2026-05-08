from __future__ import annotations

import itertools
import time
from dataclasses import replace
from typing import Iterable

import numpy as np

from .binary import run_binary_vqe
from .data import get_stock_data
from .fractional import run_fractional_vqe
from .metrics import portfolio_return, portfolio_risk
from .qaoa import run_qaoa
from .types import BinaryVQEConfig, FractionalVQEConfig, QAOAConfig


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
    """Solve the long-only simplex Markowitz problem by active-set enumeration.

    This is intended for small benchmark/client examples. It exactly solves the
    convex quadratic objective over every possible positive support, then keeps
    the best feasible KKT candidate.
    """
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


def labels_from_selection(labels: Iterable[str], x: np.ndarray) -> str:
    return ", ".join(label for label, selected in zip(labels, x) if int(selected))


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
    mu = mu_series.values
    Sigma = sigma_df.values

    def binary_row(
        method: str,
        method_type: str,
        x: np.ndarray,
        runtime: float,
        feasible_rate: float,
        notes: str,
    ) -> dict[str, str]:
        return {
            "dataset": "Real data method comparison",
            "method": method,
            "type": method_type,
            "window": f"{start} to {end}",
            "lambda": f"{lam:.6g}",
            "objective_family": "binary_qubo",
            "reported_weighting": "equal_weight_selected",
            "k": str(k),
            "selection_or_weights": labels_from_selection(tickers, x),
            "bitstring": bitstring(x),
            "return": f"{portfolio_return(mu, selected_equal_weight(x)):.6f}",
            "risk": f"{portfolio_risk(Sigma, selected_equal_weight(x)):.6f}",
            "objective": f"{binary_objective(x, mu, Sigma, lam, alpha, k):.6f}",
            "feasible_rate": f"{feasible_rate:.6f}",
            "runtime_seconds": f"{runtime:.6f}",
            "notes": notes,
        }

    def weight_row(
        method: str,
        method_type: str,
        w: np.ndarray,
        runtime: float,
        notes: str,
    ) -> dict[str, str]:
        feasible = float(np.all(w >= -1e-8) and abs(w.sum() - 1.0) < 1e-6)
        return {
            "dataset": "Real data method comparison",
            "method": method,
            "type": method_type,
            "window": f"{start} to {end}",
            "lambda": f"{lam:.6g}",
            "objective_family": "fractional_simplex",
            "reported_weighting": "simplex_weights",
            "k": "",
            "selection_or_weights": weights_string(w),
            "bitstring": "",
            "return": f"{portfolio_return(mu, w):.6f}",
            "risk": f"{portfolio_risk(Sigma, w):.6f}",
            "objective": f"{fractional_objective(w, mu, Sigma, lam):.6f}",
            "feasible_rate": f"{feasible:.6f}",
            "runtime_seconds": f"{runtime:.6f}",
            "notes": notes,
        }

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
    x_binary = np.asarray(
        (
            binary_res.x_best_feasible
            if binary_res.x_best_feasible is not None
            else binary_res.x_topk
        ),
        dtype=int,
    )
    rows.append(
        binary_row(
            "Binary VQE best feasible",
            "quantum",
            x_binary,
            time.perf_counter() - started,
            feasible_sample_rate(binary_res.sample_counts, k),
            "Falls back to Top-K projection if needed",
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
        x_qaoa = np.asarray(
            (
                qaoa_res.x_best_feasible
                if qaoa_res.x_best_feasible is not None
                else qaoa_res.x_topk
            ),
            dtype=int,
        )
        rows.append(
            binary_row(
                f"QAOA {mixer.upper()} best feasible",
                "quantum",
                x_qaoa,
                time.perf_counter() - started,
                feasible_sample_rate(qaoa_res.sample_counts, k),
                "Falls back to Top-K projection if needed",
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
            "Fractional VQE",
            "quantum",
            np.asarray(fractional_res.weights, dtype=float),
            time.perf_counter() - started,
            "Simplex-normalized allocation",
        )
    )

    return rows, prices


def seeded_config(cfg, seed: int):
    """Return a dataclass config copy with an updated seed."""
    return replace(cfg, seed=seed)
