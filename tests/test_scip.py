"""SCIP formulation, native metadata, and streaming incumbent integration."""

import numpy as np
import pytest

from vqe_portfolio.baselines import exhaustive_binary_baseline, scip_binary_baseline
from vqe_portfolio.problem import PortfolioProblem

pyscipopt = pytest.importorskip("pyscipopt")


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", [0, 1, 3, 6])
@pytest.mark.parametrize("lam", [0, 0.7])
def test_scip_matches_enumeration(seed, k, lam):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(6, 6))
    mu, sigma = rng.normal(size=6), factors @ factors.T / 6
    problem = PortfolioProblem(mu, sigma, lam, "binary_qubo", k)
    published = []

    def publish(x, details):
        assert details["source"] == "scip"
        published.append(x.copy())
        x[:] = -100  # Neither the retained incumbent nor native state can change.

    result = scip_binary_baseline(
        problem.mu, problem.Sigma, lam, k, candidate_callback=publish
    )
    _, expected = exhaustive_binary_baseline(mu, sigma, lam, 0, k)
    assert result.status == "optimal"
    assert problem.evaluate(result.candidate).feasible
    assert result.objective == pytest.approx(expected, abs=1e-7)
    assert result.lower_bound == pytest.approx(expected, abs=1e-7)
    assert result.mip_gap == pytest.approx(0, abs=1e-7)
    scores = [problem.evaluate(x) for x in published]
    assert scores and all(s.feasible for s in scores)
    assert all(a.objective > b.objective for a, b in zip(scores, scores[1:]))
    assert scores[-1].objective == pytest.approx(result.objective)


def test_scip_supports_negative_objective_and_signed_nonconvex_couplings():
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([[1, -2, 0.5], [-2, 1, 0.2], [0.5, 0.2, 1]])
    result = scip_binary_baseline(mu, sigma, 0.1, 2)
    _, expected = exhaustive_binary_baseline(mu, sigma, 0.1, 0, 2)
    assert result.status == "optimal"
    assert result.objective < 0
    assert result.objective == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw,status,bound,gap,expected_status",
    [
        (None, "timelimit", -2.0, 1e20, "timelimit"),
        ([1, 0, 0], "timelimit", -2.0, 0.5, "timelimit"),
        ([1, 0, 0], "gaplimit", -2.0, 0.5, "gaplimit"),
        ([0.5, 0.5, 0], "timelimit", -2.0, 0.5, "invalid_candidate"),
        ([1, 1, 0], "timelimit", -2.0, 0.5, "invalid_candidate"),
        ([float("nan"), 0, 0], "timelimit", -2.0, 0.5, "invalid_candidate"),
        (None, "timelimit", -1e20, 1e20, "timelimit"),
        ([1, 0, 0], "timelimit", -float("inf"), float("inf"), "timelimit"),
    ],
)
def test_scip_preserves_native_limits_and_rejects_invalid_incumbents(
    monkeypatch, raw, status, bound, gap, expected_status
):
    freed = []

    class LimitedModel(pyscipopt.Model):
        def optimize(self):
            pass

        def getStatus(self):
            return status

        def getBestSol(self):
            return None if raw is None else object()

        def getSolVal(self, solution, variable):
            return raw[int(variable.name.removeprefix("x_"))]

        def getDualbound(self):
            return bound

        def getGap(self):
            return gap

        def freeProb(self):
            freed.append(True)
            return super().freeProb()

    monkeypatch.setattr(pyscipopt, "Model", LimitedModel)
    result = scip_binary_baseline(np.array([1, 2, 3]), np.eye(3), 0.1, 1)
    assert result.status == expected_status
    assert result.lower_bound == (bound if abs(bound) < 1e20 else None)
    valid = raw is not None and expected_status != "invalid_candidate"
    assert (result.candidate is not None) == valid
    assert result.mip_gap == (gap if valid and np.isfinite(gap) else None)
    if valid:
        assert result.objective == pytest.approx(-0.9)
    else:
        assert result.objective is None
    assert freed == [True]


def test_scip_callback_errors_propagate_after_native_solver_stops():
    def failed(*args):
        raise RuntimeError("incumbent consumer failed")

    with pytest.raises(RuntimeError, match="incumbent consumer failed"):
        scip_binary_baseline(
            np.array([1, 2, 3]), np.eye(3), 0.1, 1, candidate_callback=failed
        )


def test_scip_single_asset():
    for k in [0, 1]:
        result = scip_binary_baseline(np.array([3.0]), np.array([[0.2]]), 1, k)
        assert result.status == "optimal"
        np.testing.assert_array_equal(result.candidate, [k])
