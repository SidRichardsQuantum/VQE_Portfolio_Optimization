"""
vqe_portfolio

Variational Quantum Eigensolver (VQE)–based portfolio optimization.

This package provides:
- Binary VQE for asset selection with cardinality constraints
- Fractional VQE for long-only portfolio allocation on the simplex
- Classical utilities for evaluation (efficient frontiers, metrics)
- Lightweight helpers for reproducibility and notebook workflows
"""

# ======================
# Market data utilities
# ======================
from .data import (
    get_stock_data,
    fetch_prices,
    compute_mu_sigma,
)

# ======================
# Configuration & results
# ======================
from .types import (
    BinaryVQEConfig,
    FractionalVQEConfig,
    LambdaSweepConfig,
    BinaryVQEResult,
    FractionalVQEResult,
)

# ======================
# Core algorithms
# ======================
from .binary import (
    run_binary_vqe,
    binary_lambda_sweep,
)

from .fractional import (
    run_fractional_vqe,
    fractional_lambda_sweep,
)

# ======================
# Evaluation utilities
# ======================
from .frontier import (
    Frontier,
    binary_frontier_from_probs,
    fractional_frontier_from_allocs,
)

# ======================
# Public utilities
# ======================
from .utils import (
    set_global_seed,
    resolve_notebook_outdir,
)

# ======================
# Public API
# ======================
__all__ = [
    # --- data ---
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
