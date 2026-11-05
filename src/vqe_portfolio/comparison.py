"""Compatibility exports; prefer baselines, benchmark, and research.comparison."""

from .baselines import (
    equal_weight_baseline,
    exact_long_only_markowitz_baseline,
    exhaustive_binary_baseline,
    minimum_variance_binary_baseline,
    top_return_baseline,
)
from .benchmark import (
    bitstring,
    feasible_sample_rate,
    labels_from_selection,
    selected_equal_weight,
    weights_string,
)
from .problem import (
    binary_objective,
    fractional_objective,
)
from .research.comparison import (
    real_data_comparison_rows,
)
from .utils import (
    seeded_config,
)

__all__ = [
    "binary_objective",
    "bitstring",
    "equal_weight_baseline",
    "exact_long_only_markowitz_baseline",
    "exhaustive_binary_baseline",
    "feasible_sample_rate",
    "fractional_objective",
    "labels_from_selection",
    "minimum_variance_binary_baseline",
    "real_data_comparison_rows",
    "seeded_config",
    "selected_equal_weight",
    "top_return_baseline",
    "weights_string",
]
