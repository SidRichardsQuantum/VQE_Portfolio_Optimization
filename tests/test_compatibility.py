"""Protect existing imports as implementations move into focused namespaces."""

import importlib

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "validation",
        "training_diagnostics",
        "training_effort",
        "sampling_allowance",
        "failure_diagnostics",
    ],
)
def test_legacy_research_modules_share_implementation(name):
    legacy = importlib.import_module(f"vqe_portfolio.{name}")
    canonical = importlib.import_module(f"vqe_portfolio.research.{name}")
    assert legacy is canonical


def test_classical_module_and_comparison_exports_remain_compatible():
    from vqe_portfolio import baselines, benchmark, classical, comparison, problem
    from vqe_portfolio.baselines import solvers
    from vqe_portfolio.research.comparison import real_data_comparison_rows

    assert classical is solvers
    for name in (
        "equal_weight_baseline",
        "exact_long_only_markowitz_baseline",
        "exhaustive_binary_baseline",
        "minimum_variance_binary_baseline",
        "top_return_baseline",
    ):
        assert getattr(comparison, name) is getattr(baselines, name)
    for name in (
        "bitstring",
        "weights_string",
        "feasible_sample_rate",
        "selected_equal_weight",
        "labels_from_selection",
    ):
        assert getattr(comparison, name) is getattr(benchmark, name)
    assert comparison.binary_objective is problem.binary_objective
    assert comparison.fractional_objective is problem.fractional_objective
    assert comparison.real_data_comparison_rows is real_data_comparison_rows
