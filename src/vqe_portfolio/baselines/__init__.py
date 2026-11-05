"""Classical reference, heuristic, and optional solver baselines.

Optional solver dependencies are loaded only when a solver is called.
"""

from .reference import (
    equal_weight_baseline,
    exact_long_only_markowitz_baseline,
    exhaustive_binary_baseline,
    minimum_variance_binary_baseline,
    top_return_baseline,
)
from .solvers import (
    ClassicalResult,
    convex_markowitz_baseline,
    local_search_binary_baseline,
    mixed_integer_binary_baseline,
)

__all__ = [
    "equal_weight_baseline",
    "exact_long_only_markowitz_baseline",
    "exhaustive_binary_baseline",
    "minimum_variance_binary_baseline",
    "top_return_baseline",
    "ClassicalResult",
    "convex_markowitz_baseline",
    "local_search_binary_baseline",
    "mixed_integer_binary_baseline",
]
