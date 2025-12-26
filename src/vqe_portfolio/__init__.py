"""
vqe_portfolio

Variational Quantum Eigensolver (VQE)–based portfolio optimization.

This package provides:
- Binary VQE for asset selection with cardinality constraints
- Fractional VQE for long-only portfolio allocation on the simplex
- Classical utilities for evaluation (efficient frontiers, metrics)
- Lightweight helpers for reproducibility and notebook workflows
"""

# ---------------------------------------------------------------------
# Optional data utilities (lazy import; require vqe-portfolio[data])
# ---------------------------------------------------------------------

_DATA_DEPS_ERROR = (
    "Data utilities require optional dependencies. "
    "Install with: pip install 'vqe-portfolio[data]'"
)


def _lazy_data_import(func_name: str):
    try:
        from . import data
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(_DATA_DEPS_ERROR) from e

    try:
        return getattr(data, func_name)
    except AttributeError as e:
        # Should never happen unless the module is corrupted
        raise RuntimeError(f"Missing expected function data.{func_name}") from e


def get_stock_data(*args, **kwargs):
    return _lazy_data_import("get_stock_data")(*args, **kwargs)


def fetch_prices(*args, **kwargs):
    return _lazy_data_import("fetch_prices")(*args, **kwargs)


def compute_mu_sigma(*args, **kwargs):
    return _lazy_data_import("compute_mu_sigma")(*args, **kwargs)


# ---------------------------------------------------------------------
# Configuration & result types
# ---------------------------------------------------------------------

from .types import (
    BinaryVQEConfig,
    FractionalVQEConfig,
    LambdaSweepConfig,
    BinaryVQEResult,
    FractionalVQEResult,
)

# ---------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------

from .binary import (
    run_binary_vqe,
    binary_lambda_sweep,
)

from .fractional import (
    run_fractional_vqe,
    fractional_lambda_sweep,
)

# ---------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------

from .frontier import (
    Frontier,
    binary_frontier_from_probs,
    fractional_frontier_from_allocs,
)

# ---------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------

from .utils import (
    set_global_seed,
    resolve_notebook_outdir,
)

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    # --- data (optional extras) ---
    "get_stock_data",
    "fetch_prices",
    "compute_mu_sigma",

    # --- configs & results ---
    "BinaryVQEConfig",
    "FractionalVQEConfig",
    "LambdaSweepConfig",
    "BinaryVQEResult",
    "FractionalVQEResult",

    # --- algorithms ---
    "run_binary_vqe",
    "binary_lambda_sweep",
    "run_fractional_vqe",
    "fractional_lambda_sweep",

    # --- evaluation ---
    "Frontier",
    "binary_frontier_from_probs",
    "fractional_frontier_from_allocs",

    # --- utilities ---
    "set_global_seed",
    "resolve_notebook_outdir",
]
