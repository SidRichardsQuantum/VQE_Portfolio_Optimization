# src/vqe_portfolio/cli.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def _to_jsonable(x: Any) -> Any:
    """Convert common scientific/Python objects into JSON-serializable types."""
    if is_dataclass(x):
        return {k: _to_jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def _read_input_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object/dict at the top level.")
    return data


def _parse_vector_csv(s: str) -> np.ndarray:
    # "0.1,0.2,0.3"
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty vector.")
    return np.asarray(vals, dtype=float)


def _parse_matrix_rows(s: str) -> np.ndarray:
    # "1,0.1;0.1,2"  (rows separated by ';')
    rows = []
    for row in s.split(";"):
        row = row.strip()
        if not row:
            continue
        rows.append([float(x.strip()) for x in row.split(",") if x.strip()])
    if not rows:
        raise ValueError("Empty matrix.")
    mat = np.asarray(rows, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Sigma must be a square matrix.")
    return mat


def _parse_tickers(s: str) -> list[str]:
    t = [x.strip().upper() for x in s.split(",") if x.strip()]
    if not t:
        raise ValueError("No tickers provided.")
    return t


def _load_mu_sigma_from_market(args: argparse.Namespace):
    # Optional dependency gate: give a clean error message.
    try:
        from vqe_portfolio.data import get_stock_data
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Real market data requires optional dependencies. "
            "Install with: pip install 'vqe-portfolio[data]'"
        ) from e

    tickers = _parse_tickers(args.tickers)
    mu, Sigma, prices = get_stock_data(
        tickers,
        start=args.start,
        end=args.end,
        auto_adjust=not args.no_auto_adjust,
        use_log=not args.simple_returns,
        shrink=args.shrink,
        scale=args.scale,
        progress=args.progress
    )

    # Return numpy arrays to match VQE API expectations.
    return mu.values, Sigma.values, prices, tickers


def _load_mu_sigma(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if args.input is not None:
        payload = _read_input_json(Path(args.input))
        if "mu" not in payload or "sigma" not in payload:
            raise ValueError("Input JSON must contain keys: 'mu' and 'sigma'.")
        mu = np.asarray(payload["mu"], dtype=float)
        sigma = np.asarray(payload["sigma"], dtype=float)
        return mu, sigma

    if args.mu is None or args.sigma is None:
        raise ValueError("Provide either --input JSON or both --mu and --sigma.")

    mu = _parse_vector_csv(args.mu)
    sigma = _parse_matrix_rows(args.sigma)

    if sigma.shape[0] != mu.shape[0]:
        raise ValueError(f"Dimension mismatch: len(mu)={mu.shape[0]} vs sigma={sigma.shape}.")
    return mu, sigma


def _write_output(out_path: Optional[str], payload: Dict[str, Any]) -> None:
    text = json.dumps(_to_jsonable(payload), indent=2, sort_keys=True)
    if out_path:
        Path(out_path).write_text(text)
    else:
        print(text)


def _cmd_binary(args: argparse.Namespace) -> int:
    from vqe_portfolio.binary import run_binary_vqe
    from vqe_portfolio.types import BinaryVQEConfig

    mu, Sigma = _load_mu_sigma(args)

    cfg = BinaryVQEConfig(
        depth=args.depth,
        steps=args.steps,
        stepsize=args.stepsize,
        shots_train=args.shots_train,
        shots_sample=args.shots_sample,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        # required problem params
        lam=args.lam,
        alpha=args.alpha,
        k=args.k,
    )

    res = run_binary_vqe(mu=mu, Sigma=Sigma, cfg=cfg)

    payload = {
        "method": "binary",
        "config": cfg,
        "result": res,
    }
    _write_output(args.out, payload)
    return 0


def _cmd_fractional(args: argparse.Namespace) -> int:
    from vqe_portfolio.fractional import run_fractional_vqe
    from vqe_portfolio.types import FractionalVQEConfig

    mu, Sigma = _load_mu_sigma(args)

    cfg = FractionalVQEConfig(
        steps=args.steps,
        stepsize=args.stepsize,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        shots=args.shots,
        lam=args.lam,
    )

    res = run_fractional_vqe(mu=mu, Sigma=Sigma, cfg=cfg)

    payload = {
        "method": "fractional",
        "config": cfg,
        "result": res,
    }
    _write_output(args.out, payload)
    return 0


def _cmd_binary_data(args: argparse.Namespace) -> int:
    from vqe_portfolio.binary import run_binary_vqe
    from vqe_portfolio.types import BinaryVQEConfig

    mu, Sigma, prices, tickers = _load_mu_sigma_from_market(args)

    cfg = BinaryVQEConfig(
        depth=args.depth,
        steps=args.steps,
        stepsize=args.stepsize,
        log_every=args.log_every,
        lam=args.lam,
        alpha=args.alpha,
        k=args.k,
        device=args.device,
        shots_train=args.shots_train,
        shots_sample=args.shots_sample,
        seed=args.seed,
    )

    res = run_binary_vqe(mu=mu, Sigma=Sigma, cfg=cfg)

    payload = {
        "method": "binary",
        "mode": "market-data",
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "data_options": {
            "auto_adjust": not args.no_auto_adjust,
            "use_log": not args.simple_returns,
            "shrink": args.shrink,
            "scale": args.scale,
        },
        "config": cfg,
        "result": res,
    }

    # Optional: include mu/Sigma for reproducibility
    if args.include_mu_sigma:
        payload["mu"] = mu
        payload["sigma"] = Sigma

    _write_output(args.out, payload)

    # Optional: persist prices CSV
    if args.prices_csv:
        prices.to_csv(args.prices_csv)

    return 0


def _cmd_fractional_data(args: argparse.Namespace) -> int:
    from vqe_portfolio.fractional import run_fractional_vqe
    from vqe_portfolio.types import FractionalVQEConfig

    mu, Sigma, prices, tickers = _load_mu_sigma_from_market(args)

    cfg = FractionalVQEConfig(
        steps=args.steps,
        stepsize=args.stepsize,
        log_every=args.log_every,
        lam=args.lam,
        device=args.device,
        shots=args.shots,
        seed=args.seed,
    )

    res = run_fractional_vqe(mu=mu, Sigma=Sigma, cfg=cfg)

    payload = {
        "method": "fractional",
        "mode": "market-data",
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "data_options": {
            "auto_adjust": not args.no_auto_adjust,
            "use_log": not args.simple_returns,
            "shrink": args.shrink,
            "scale": args.scale,
        },
        "config": cfg,
        "result": res,
    }

    if args.include_mu_sigma:
        payload["mu"] = mu
        payload["sigma"] = Sigma

    _write_output(args.out, payload)

    if args.prices_csv:
        prices.to_csv(args.prices_csv)

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vqe-portfolio",
        description="VQE-based portfolio optimization (binary selection and fractional allocation).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_output_only(sp: argparse.ArgumentParser) -> None:
        o = sp.add_argument_group("output")
        o.add_argument("--out", type=str, default=None, help="Write JSON output to this path (default: stdout).")

    def add_common_io(sp: argparse.ArgumentParser) -> None:
        g = sp.add_argument_group("input")
        g.add_argument("--input", type=str, default=None, help="Path to JSON containing mu and sigma.")
        g.add_argument("--mu", type=str, default=None, help="CSV vector, e.g. '0.1,0.2,0.05'.")
        g.add_argument(
            "--sigma",
            type=str,
            default=None,
            help="Row-separated matrix, e.g. '1,0.1;0.1,2'. Use ';' between rows.",
        )
        add_output_only(sp)

    def add_common_binary_vqe(sp: argparse.ArgumentParser) -> None:
        g = sp.add_argument_group("binary-vqe")
        g.add_argument("--depth", type=int, default=2)
        g.add_argument("--steps", type=int, default=100)
        g.add_argument("--stepsize", type=float, default=0.25)
        g.add_argument("--seed", type=int, default=0)
        g.add_argument("--device", type=str, default="default.qubit")
        g.add_argument("--shots-train", type=int, default=None)
        g.add_argument("--shots-sample", type=int, default=2000)
        g.add_argument("--log-every", type=int, default=10)

    def add_common_fractional_vqe(sp: argparse.ArgumentParser) -> None:
        g = sp.add_argument_group("fractional-vqe")
        g.add_argument("--steps", type=int, default=100)
        g.add_argument("--stepsize", type=float, default=0.25)
        g.add_argument("--seed", type=int, default=0)
        g.add_argument("--device", type=str, default="default.qubit")
        g.add_argument("--shots", type=int, default=None)
        g.add_argument("--log-every", type=int, default=10)

    def add_market_data_args(sp: argparse.ArgumentParser) -> None:
        g = sp.add_argument_group("market-data")
        g.add_argument("--tickers", type=str, required=True, help='Comma-separated, e.g. "AAPL,MSFT,NVDA".')
        g.add_argument("--start", type=str, default="2023-01-01", help="Start date YYYY-MM-DD.")
        g.add_argument("--end", type=str, default="2024-01-01", help="End date YYYY-MM-DD.")
        g.add_argument("--no-auto-adjust", action="store_true", help="Disable yfinance auto_adjust.")
        g.add_argument("--simple-returns", action="store_true", help="Use simple returns instead of log returns.")
        g.add_argument("--shrink", type=str, default=None, choices=["lw"], help="Covariance shrinkage. Use 'lw' for Ledoit-Wolf; omit for sample covariance.",)
        g.add_argument("--scale", type=str, default="none", choices=["none", "trace", "max"], help="Covariance scaling.")
        g.add_argument("--include-mu-sigma", action="store_true", help="Include mu/sigma arrays in JSON output.")
        g.add_argument("--progress", action="store_true", help="Show yfinance download progress.")
        g.add_argument("--prices-csv", type=str, default=None, help="Optional path to save the fetched prices CSV.")

    # -----------------------
    # binary (synthetic / user-provided mu,sigma)
    # -----------------------
    sp_b = sub.add_parser("binary", help="Run Binary VQE (asset selection under cardinality constraint).")
    add_common_io(sp_b)
    add_common_binary_vqe(sp_b)
    sp_b.add_argument("--k", type=int, required=True, help="Cardinality constraint (cfg.k).")
    sp_b.add_argument("--lam", type=float, required=True, help="Risk aversion parameter (cfg.lam).")
    sp_b.add_argument("--alpha", type=float, default=10.0, help="Penalty weight (cfg.alpha).")
    sp_b.set_defaults(func=_cmd_binary)

    # -----------------------
    # fractional (synthetic / user-provided mu,sigma)
    # -----------------------
    sp_f = sub.add_parser("fractional", help="Run Fractional VQE (long-only allocation on the simplex).")
    add_common_io(sp_f)
    add_common_fractional_vqe(sp_f)
    sp_f.add_argument("--lam", type=float, required=True, help="Risk aversion parameter (cfg.lam).")
    sp_f.set_defaults(func=_cmd_fractional)

    # -----------------------
    # market data
    # -----------------------
    sp_bd = sub.add_parser(
        "binary-data",
        help="Run Binary VQE using real market data (requires vqe-portfolio[data]).",
    )
    add_output_only(sp_bd)
    add_common_binary_vqe(sp_bd)
    add_market_data_args(sp_bd)
    sp_bd.add_argument("--k", type=int, required=True)
    sp_bd.add_argument("--lam", type=float, required=True)
    sp_bd.add_argument("--alpha", type=float, default=10.0)
    sp_bd.set_defaults(func=_cmd_binary_data)

    sp_fd = sub.add_parser(
        "fractional-data",
        help="Run Fractional VQE using real market data (requires vqe-portfolio[data]).",
    )
    add_output_only(sp_fd)
    add_common_fractional_vqe(sp_fd)
    add_market_data_args(sp_fd)
    sp_fd.add_argument("--lam", type=float, required=True)
    sp_fd.set_defaults(func=_cmd_fractional_data)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))
