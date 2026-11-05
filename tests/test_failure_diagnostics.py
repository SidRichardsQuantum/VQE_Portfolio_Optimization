import itertools
import json

import numpy as np
import pytest

from vqe_portfolio.problem import PortfolioProblem
from vqe_portfolio.qaoa import xy_mixer_edges
from vqe_portfolio.research import failure_diagnostics as f
from vqe_portfolio.research.training_effort import measure_trajectory
from vqe_portfolio.research.validation import (
    ValidationConfig,
    prepare_instances,
    trial_plan,
)


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
def test_exact_laurent_amplitudes_match_production_xy_circuit(n):
    k = n // 2
    problem = PortfolioProblem(
        np.linspace(0.1, 0.2, n), np.eye(n) * 0.02, 4, "binary_qubo", k
    )
    polynomials = f.xy_support_polynomials(n, k)
    for beta in (0, 0.173, 0.51, np.pi / 4, np.pi / 2, np.pi):
        exact = f.polynomial_probabilities(n, polynomials, beta)
        actual = f.circuit_probabilities(problem, "qaoa-xy", [[0.37, beta]], 1, 5)
        np.testing.assert_allclose(exact, actual, atol=1e-12)
        assert exact.sum() == pytest.approx(1)
    if n == 8:
        assert set(polynomials) == {"11110000", "11101000", "01110001", "01101001"}


def test_gate_order_bounds_and_gamma_roles():
    assert xy_mixer_edges(8) == (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 0),
    )
    assert [len(f.xy_support_upper_bound(8, 4, p)) for p in (1, 2, 3)] == [4, 66, 70]
    problem = PortfolioProblem(
        np.linspace(0.1, 0.2, 8), np.eye(8) * 0.02, 4, "binary_qubo", 4
    )
    for depth in (1, 2, 3):
        params = np.random.default_rng(314 + depth).uniform(0.2, 1.1, (depth, 2))
        before = f.circuit_probabilities(problem, "qaoa-xy", params, depth, 5)
        shifted = params.copy()
        shifted[0, 0] += 0.7
        np.testing.assert_allclose(
            f.circuit_probabilities(problem, "qaoa-xy", shifted, depth, 5),
            before,
            atol=1e-12,
        )
        bound = f.xy_support_upper_bound(8, 4, depth)
        outside = [i for i in range(256) if np.binary_repr(i, width=8) not in bound]
        assert before[outside].sum() < 1e-20
    # A later cost angle affects a generic p=2 distribution with correlated data.
    rng = np.random.default_rng(19)
    factors = rng.normal(size=(8, 3)) * 0.1
    problem = PortfolioProblem(
        np.linspace(0.1, 0.2, 8),
        factors @ factors.T + np.eye(8) * 0.02,
        4,
        "binary_qubo",
        4,
    )
    params = np.array([[0.4, 0.3], [0.7, 0.2]])
    changed = params.copy()
    changed[1, 0] += 0.3
    assert (
        np.linalg.norm(
            f.circuit_probabilities(problem, "qaoa-xy", params, 2, 5)
            - f.circuit_probabilities(problem, "qaoa-xy", changed, 2, 5)
        )
        > 1e-5
    )


def test_penalty_threshold_and_ties_are_distinct_from_training_success():
    problem = PortfolioProblem([0.1, 0.2], np.zeros((2, 2)), 1, "binary_qubo", 1)
    rows = f.penalty_diagnostic(problem, -0.2, [0.05, 0.1, 0.5])
    assert rows[0]["critical_alpha"] == pytest.approx(0.1)
    assert not rows[0]["all_numerical_minimisers_feasible"]
    assert not rows[1]["all_numerical_minimisers_feasible"]
    assert rows[2]["strict_sufficient"] and rows[2]["all_numerical_minimisers_feasible"]


def test_vqe_can_represent_every_basis_portfolio_at_both_depths():
    problem = PortfolioProblem([0.1, 0.3, 0.2], np.eye(3) * 0.01, 1, "binary_qubo", 1)
    for bits in itertools.product((0, 1), repeat=3):
        for depth in (2, 3):
            params = np.zeros((depth, 3))
            params[0] = np.pi * np.array(bits)
            p = f.circuit_probabilities(problem, "binary-vqe", params, depth, 5)
            assert p[int("".join(map(str, bits)), 2)] == pytest.approx(1)


def test_mode_gradient_swap_and_product_path_energy():
    problem = PortfolioProblem([0.1, 0.3, 0.2], np.eye(3) * 0.01, 1, "binary_qubo", 1)
    x = np.array([1, 0, 0])
    neighbour = np.array([0, 1, 0])
    alpha = 5
    params = np.zeros((2, 3))
    params[0] = np.pi * x
    p = f.circuit_probabilities(problem, "binary-vqe", params, 2, alpha)
    mode = f.vqe_mode_diagnostic(problem, params, p, 2, alpha, -0.29)
    assert mode["mode_feasible"] and mode["best_swap"] == "010"
    assert mode["swap_improvement"] == pytest.approx(0.2)
    assert mode["final_gradient_norm"] < 1e-12
    path = f.swap_path_barrier(problem, x, neighbour, alpha)
    assert path["barrier"] > 2
    energies = []
    bits = np.array(list(itertools.product((0, 1), repeat=3)))
    diagonal = np.array(
        [
            b @ problem.Sigma @ b - problem.mu @ b + alpha * (b.sum() - 1) ** 2
            for b in bits
        ]
    )
    for t in np.linspace(0, 1, 21):
        theta = params.copy()
        theta[0, 0] = np.pi * (1 - t)
        theta[0, 1] = np.pi * t
        actual = (
            f.circuit_probabilities(problem, "binary-vqe", theta, 2, alpha) @ diagonal
        )
        u = np.sin(np.pi * t / 2) ** 2
        predicted = (
            path["initial_objective"] + path["linear"] * u + path["quadratic"] * u * u
        )
        assert actual == pytest.approx(predicted)
        energies.append(actual - path["initial_objective"])
    assert max(energies) <= path["barrier"] + 1e-12
    invalid = f.vqe_mode_diagnostic(
        problem, np.zeros((2, 3)), [1, 0, 0, 0, 0, 0, 0, 0], 2, 5, -0.29
    )
    assert not invalid["mode_feasible"] and invalid["mode_objective"] is None


def test_runner_reuses_baselines_and_resumes_variants(tmp_path, monkeypatch):
    from scripts import run_failure_diagnostics as runner

    config = ValidationConfig(
        asset_counts=(3,),
        instance_seeds=(11,),
        solver_seeds=(0,),
        steps=2,
        shots=16,
        methods=("binary-vqe", "qaoa-x", "qaoa-xy"),
    )
    instances = prepare_instances(config)
    trials = [
        measure_trajectory(
            key, instance, method, seed, config, (0, 1, 2), (16, 64, 512)
        )
        for key, instance, method, seed in trial_plan(instances, config)
    ]
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study": "continuous_training_effort",
                "config": config.to_dict(),
                "instances": instances,
                "checkpoint_steps": [0, 1, 2],
                "draw_counts": [16, 64, 512],
            }
        )
    )
    (source / "trajectories.json").write_text(json.dumps(trials))
    output = tmp_path / "study"
    args = [
        "--effort-dir",
        str(source),
        "--output-dir",
        str(output),
        "--assets",
        "3",
        "--xy-depths",
        "1,2",
        "--vqe-depths",
        "2",
        "--penalties",
        "0.5,5",
        "--beta-points",
        "9",
    ]
    assert runner.main(args) == 0
    records = json.loads((output / "trajectories.json").read_text())
    assert len(records) == 4 and sum(r["reused_baseline"] for r in records) == 2
    assert all(not r["error"] for r in records)
    assert (output / "xy_support.png").exists() and (
        output / "failure_sensitivity.png"
    ).exists()
    (output / "trajectories.json").write_text(json.dumps(records[:3]))
    original = runner.measure_variant
    calls = []

    def record(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(runner, "measure_variant", record)
    assert runner.main(args + ["--resume"]) == 0
    assert calls == [records[-1]["trial_id"]]
    assert json.loads((output / "trajectories.json").read_text())[:3] == records[:3]
    with pytest.raises(ValueError, match="nonempty"):
        runner.main(args)
    with pytest.raises(ValueError, match="Resume refused"):
        runner.main(args + ["--penalties", "0.1,5", "--resume"])
