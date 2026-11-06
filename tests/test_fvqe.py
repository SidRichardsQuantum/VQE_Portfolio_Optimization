"""Independent filtered-state derivatives and end-to-end F-VQE checks."""

from dataclasses import replace
import itertools
import json

import numpy as np
import pennylane as qml
import pytest

from vqe_portfolio import FVQEConfig, FVQEResult, PortfolioProblem, run_fvqe
from vqe_portfolio.ansatz import (
    BINARY_ANSATZES,
    apply_binary_ansatz,
    binary_ansatz_shape,
)
from vqe_portfolio.binary import build_ising_hamiltonian
from vqe_portfolio.cli import main
from vqe_portfolio.fvqe import (
    _choose_tau,
    _filter_gradient,
    _scaled_energies,
    _spectrum_bounds,
)


MU = np.array([0.1, 0.2, 0.15])
SIGMA = np.array([[0.05, 0.01, 0], [0.01, 0.06, 0.01], [0, 0.01, 0.04]])


def state_function(ansatz, depth, n):
    @qml.qnode(qml.device("default.qubit", wires=n))
    def state(theta):
        apply_binary_ansatz(ansatz, theta, depth, n)
        return qml.state()

    return state


def finite_gradient(cost, theta):
    gradient = np.zeros_like(theta)
    for index in np.ndindex(theta.shape):
        plus, minus = theta.copy(), theta.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        gradient[index] = (cost(plus) - cost(minus)) / 2e-6
    return gradient


@pytest.mark.parametrize("ansatz", BINARY_ANSATZES)
@pytest.mark.parametrize("filter_type", ["inverse", "exponential"])
def test_gradient_matches_independent_frozen_filtered_state_distance(
    ansatz, filter_type
):
    """Checks phase-sensitive overlap, normalization, shift factor and sign."""
    shape = binary_ansatz_shape(ansatz, 2, 3)
    theta = np.random.default_rng(5).uniform(0.3, 2.2, shape)
    state = state_function(ansatz, 2, 3)
    h = build_ising_hamiltonian(MU, SIGMA, 2, 1, 2)
    energies = np.diag(qml.matrix(h, wire_order=range(3))).real
    lo, hi = _spectrum_bounds(MU, SIGMA, 2, 1, 2)
    normalized = _scaled_energies(energies, lo, hi, 0.001)
    tau = 0.7
    weights = (
        normalized ** (-tau) if filter_type == "inverse" else np.exp(-tau * normalized)
    )
    # Freeze the *state* from the previous iteration, including its phases.
    target = weights * state(theta)
    target /= np.linalg.norm(target)

    def cost(angles):
        return 1 - np.vdot(target, state(angles)).real

    expected = finite_gradient(cost, theta)
    plus, minus = [], []
    for index in np.ndindex(shape):
        shifted = theta.copy()
        shifted[index] += np.pi / 2
        plus.append((normalized, abs(state(shifted)) ** 2))
        shifted[index] -= np.pi
        minus.append((normalized, abs(state(shifted)) ** 2))
    actual = _filter_gradient(
        (normalized, abs(state(theta)) ** 2), plus, minus, filter_type, tau
    )
    np.testing.assert_allclose(actual.reshape(shape), expected, atol=2e-9)
    assert np.linalg.norm(actual) > 1e-3


@pytest.mark.parametrize("filter_type", ["inverse", "exponential"])
def test_solver_takes_one_fresh_target_distance_step_each_iteration(filter_type):
    cfg = FVQEConfig(
        k=2,
        lam=2,
        alpha=1,
        depth=1,
        steps=3,
        stepsize=0.2,
        tau=0.7,
        adapt_tau=False,
        filter_type=filter_type,
        shots_sample=16,
    )
    result = run_fvqe(MU, SIGMA, cfg)
    state = state_function(cfg.ansatz, cfg.depth, 3)
    theta = np.random.default_rng(cfg.seed).uniform(0, np.pi, result.params.shape)
    h = build_ising_hamiltonian(MU, SIGMA, cfg.lam, cfg.alpha, cfg.k)
    raw = np.diag(qml.matrix(h, wire_order=range(3))).real
    scaled = _scaled_energies(
        raw,
        result.spectrum_lower_bound,
        result.spectrum_upper_bound,
        cfg.spectrum_floor,
    )
    weights = (
        scaled ** (-cfg.tau) if filter_type == "inverse" else np.exp(-cfg.tau * scaled)
    )
    for _ in range(cfg.steps):
        target = weights * state(theta)
        target /= np.linalg.norm(target)
        gradient = finite_gradient(
            lambda angles: 1 - np.vdot(target, state(angles)).real, theta
        )
        theta -= cfg.stepsize * gradient
    np.testing.assert_allclose(result.params, theta, atol=2e-9)
    assert result.energy_trace.values[-1] == pytest.approx(abs(state(theta)) ** 2 @ raw)
    assert result.tau_trace.values == [cfg.tau] * cfg.steps
    assert result.training_objective == "filtered_state_distance"
    assert result.training_optimizer == "filtering_gradient_descent"


def test_coefficient_bounds_enclose_all_penalized_costs_without_a_reference():
    mu = np.array([0.2, -0.1, 0.8])
    sigma = np.array([[0.1, -0.02, 0.03], [-0.02, 0.2, 0], [0.03, 0, 0.3]])
    problem = PortfolioProblem(mu, sigma, 2, "binary_qubo", 1)
    lo, hi = _spectrum_bounds(mu, sigma, 2, 0.8, 1)
    costs = [
        problem.evaluate(x, alpha=0.8).penalized_energy
        for x in itertools.product((0, 1), repeat=3)
    ]
    assert all(lo - 1e-12 <= cost <= hi + 1e-12 for cost in costs)
    # Bounds are deliberately loose, not an enumerated extremal energy.
    assert lo < min(costs) and hi > max(costs)


def test_adaptive_tau_brackets_threshold_and_caps_unreachable_targets():
    cfg = FVQEConfig(tau=0.1, gradient_target=0.2, tau_max=8)
    tau, gradient = _choose_tau(lambda t: np.array([t / (1 + t)]), cfg)
    assert tau == pytest.approx(0.25, abs=1e-6)
    assert 0.19999 < np.linalg.norm(gradient) <= cfg.gradient_target
    tau, gradient = _choose_tau(lambda t: np.array([0.01 * t]), cfg)
    assert tau == cfg.tau_max and gradient[0] == pytest.approx(0.08)
    # Nonmonotone unbracketed norms select the closest tried value below target.
    tau, _ = _choose_tau(lambda t: np.array([0.1 * np.exp(-((t - 0.4) ** 2))]), cfg)
    assert tau == 0.4


@pytest.mark.parametrize("shots", [None, 32])
def test_adaptive_training_is_seeded_and_callbacks_do_not_change_training(shots):
    cfg = FVQEConfig(
        k=2, depth=1, steps=3, log_every=2, shots_train=shots, shots_sample=32
    )
    first = run_fvqe(MU, SIGMA, cfg)
    checkpoints, phases = [], []
    second = run_fvqe(
        MU,
        SIGMA,
        cfg,
        candidate_callback=lambda x, info: checkpoints.append((x, info)),
        phase_callback=phases.append,
    )
    np.testing.assert_allclose(first.params, second.params, atol=1e-12)
    assert first.energy_trace == second.energy_trace
    assert first.tau_trace == second.tau_trace
    repeat = run_fvqe(MU, SIGMA, cfg)
    assert first.sample_counts == repeat.sample_counts
    assert first.energy_trace.steps == [0, 2, 3]
    assert first.tau_trace.steps == [1, 2, 3]
    assert all(
        g <= cfg.gradient_target + 1e-12 for g in first.gradient_norm_trace.values
    )
    assert [info["step"] for _, info in checkpoints] == [0, 2, 3]
    problem = PortfolioProblem(MU, SIGMA, cfg.lam, "binary_qubo", 2)
    assert all(problem.evaluate(x).feasible for x, _ in checkpoints)
    assert checkpoints[-1][1]["final"]
    assert phases[-1] == "candidate_processing" and "optimization" in phases
    assert first.x_topk.sum() == 2
    assert list(first.sample_counts) == sorted(first.sample_counts)
    assert sum(first.sample_counts.values()) == cfg.shots_sample
    if first.x_best_feasible is not None:
        feasible = [
            np.array(list(map(int, bs)))
            for bs in first.sample_counts
            if bs.count("1") == 2
        ]
        assert problem.evaluate(first.x_best_feasible).objective == min(
            problem.evaluate(x).objective for x in feasible
        )
    if shots is not None:
        marginals = (
            sum(
                np.array(list(map(int, bs))) * count
                for bs, count in first.sample_counts.items()
            )
            / cfg.shots_sample
        )
        np.testing.assert_allclose(first.x_prob, marginals)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("shots", [None, 16])
def test_circuit_counts_include_updates_and_final_state_but_not_tau_search(
    monkeypatch, adaptive, shots
):
    import vqe_portfolio.fvqe as module

    device = qml.device
    trackers = []

    def tracked_device(*args, **kwargs):
        dev = device(*args, **kwargs)
        tracker = qml.Tracker(dev, persistent=True)
        tracker.__enter__()
        trackers.append(tracker)
        return dev

    monkeypatch.setattr(module.qml, "device", tracked_device)
    cfg = FVQEConfig(
        k=2, depth=1, steps=2, adapt_tau=adaptive, shots_train=shots, shots_sample=16
    )
    result = run_fvqe(MU, SIGMA, cfg)
    assert result.training_circuit_evaluations == 2 * (2 * result.params.size + 1) + 1
    assert trackers[0].totals["executions"] == result.training_circuit_evaluations
    assert trackers[1].totals["executions"] == 1


def test_sampled_training_uses_only_measured_bitstrings(monkeypatch):
    from types import SimpleNamespace

    import vqe_portfolio.fvqe as module

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Finite-shot F-VQE must not enumerate probabilities or states"
        )

    # Restrict the solver's API, leaving simulator internals free to simulate.
    monkeypatch.setattr(
        module,
        "qml",
        SimpleNamespace(
            device=qml.device,
            qnode=qml.qnode,
            sample=qml.sample,
            set_shots=qml.set_shots,
            probs=forbidden,
            state=forbidden,
            matrix=forbidden,
        ),
    )
    result = run_fvqe(
        MU, SIGMA, FVQEConfig(k=2, depth=1, steps=1, shots_train=1, shots_sample=1)
    )
    assert isinstance(result, FVQEResult)
    assert result.x_topk.sum() == 2


@pytest.mark.parametrize("filter_type", ["inverse", "exponential"])
def test_sampled_update_agrees_with_exact_update_with_enough_shots(filter_type):
    cfg = FVQEConfig(
        k=2,
        depth=1,
        steps=1,
        tau=0.5,
        adapt_tau=False,
        filter_type=filter_type,
        shots_sample=8,
    )
    exact = run_fvqe(MU, SIGMA, cfg)
    sampled = run_fvqe(MU, SIGMA, replace(cfg, shots_train=100000))
    np.testing.assert_allclose(sampled.params, exact.params, atol=0.003)


@pytest.mark.parametrize("shots", [None, 1])
def test_constant_hamiltonian_zero_steps_and_one_asset(shots):
    cfg = FVQEConfig(
        k=1, alpha=0, lam=0, depth=1, steps=2, shots_train=shots, shots_sample=1
    )
    initial = run_fvqe([0], [[0]], replace(cfg, steps=0))
    trained = run_fvqe([0], [[0]], cfg)
    np.testing.assert_allclose(initial.params, trained.params)
    assert trained.energy_trace.values == [0, 0]
    assert initial.energy_trace.steps == [0]
    assert initial.tau_trace.values == []
    assert trained.x_topk.tolist() == [1]
    assert all(np.isfinite(trained.gradient_norm_trace.values))


def test_large_filter_strength_is_evaluated_in_log_space():
    center = (np.array([0.001, 1.0]), np.array([0.2, 0.8]))
    plus = [(center[0], np.array([0.3, 0.7]))]
    minus = [(center[0], np.array([0.1, 0.9]))]
    # Raw inverse filter values overflow at tau=1000; the ratio is well defined.
    g = _filter_gradient(center, plus, minus, "inverse", 1000)
    assert g[0] == pytest.approx(-0.2 / (4 * np.sqrt(0.2)))
    np.testing.assert_array_equal(
        _filter_gradient(center, plus, minus, "inverse", 0), [0]
    )


@pytest.mark.parametrize(
    "options",
    [
        {"depth": 0},
        {"steps": -1},
        {"steps": True},
        {"log_every": 0},
        {"shots_train": 0},
        {"shots_train": True},
        {"shots_sample": 0},
        {"tau": 0},
        {"tau": float("nan")},
        {"tau_max": 0.5},
        {"stepsize": float("inf")},
        {"gradient_target": -1},
        {"spectrum_floor": 0},
        {"spectrum_floor": 1},
        {"tau_search_steps": 0},
        {"filter_type": "cvar"},
        {"adapt_tau": "yes"},
        {"ansatz": "bad"},
        {"seed": -1},
        {"k": 0},
        {"k": 4},
        {"k": True},
        {"alpha": -1},
    ],
)
def test_invalid_options_are_rejected_before_execution(options):
    with pytest.raises(ValueError):
        run_fvqe(MU, SIGMA, replace(FVQEConfig(k=2, steps=0), **options))


@pytest.mark.parametrize("filter_type", ["inverse", "exponential"])
def test_cli_reports_filtering_metadata_and_accepts_finite_shots(tmp_path, filter_type):
    output = tmp_path / "fvqe.json"
    assert (
        main(
            [
                "f-vqe",
                "--mu",
                "0.1,0.2,0.15",
                "--sigma",
                "0.05,0,0;0,0.06,0;0,0,0.04",
                "--k",
                "2",
                "--lam",
                "2",
                "--filter",
                filter_type,
                "--fixed-tau",
                "--tau",
                "0.5",
                "--shots-train",
                "16",
                "--shots-sample",
                "16",
                "--steps",
                "1",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["method"] == "f-vqe"
    assert payload["config"]["filter_type"] == filter_type
    assert payload["result"]["tau_trace"]["values"] == [0.5]
    assert payload["result"]["training_objective"] == "filtered_state_distance"


def test_comparison_adapter_supports_auto_penalty_and_independent_scoring():
    from vqe_portfolio.research.comparison import add_fvqe_row

    rows, runs = [], []
    cfg = FVQEConfig(k=2, steps=1, shots_sample=8)
    add_fvqe_row(rows, runs, "toy", MU, SIGMA, cfg, [7, 8])
    assert len(rows) == 1 and len(runs) == 2
    assert rows[0]["method"] == "F-VQE inverse best feasible"
    assert all(row["feasible_rate"] == "1.000000" for row in runs)
    problem = PortfolioProblem(MU, SIGMA, cfg.lam, "binary_qubo", 2)
    for row in runs:
        assert float(row["objective"]) == pytest.approx(
            problem.evaluate(list(map(int, row["selection_or_weights"]))).objective
        )


def test_market_cli_preserves_data_options_and_exports_inputs(tmp_path, monkeypatch):
    from vqe_portfolio import cli
    from types import SimpleNamespace

    saved = []
    monkeypatch.setattr(
        cli,
        "_load_mu_sigma_from_market",
        lambda args: (MU, SIGMA, SimpleNamespace(to_csv=saved.append), ["A", "B", "C"]),
    )
    output = tmp_path / "market.json"
    assert (
        main(
            [
                "f-vqe-data",
                "--tickers",
                "A,B,C",
                "--k",
                "2",
                "--lam",
                "2",
                "--steps",
                "0",
                "--shots-sample",
                "8",
                "--include-mu-sigma",
                "--prices-csv",
                str(tmp_path / "prices.csv"),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["mode"] == "market-data"
    assert payload["tickers"] == ["A", "B", "C"]
    np.testing.assert_allclose(payload["mu"], MU)
    np.testing.assert_allclose(payload["sigma"], SIGMA)
    assert payload["data_options"]["auto_adjust"]
    assert saved == [str(tmp_path / "prices.csv")]


def test_comparison_script_exposes_fvqe_and_records_seed_trials(tmp_path):
    import csv
    from scripts.generate_comparison_results import main, parse_methods

    output, trials = tmp_path / "summary.csv", tmp_path / "trials.csv"
    assert "f-vqe" in parse_methods("all")
    assert (
        main(
            [
                "--methods",
                "f-vqe",
                "--asset-counts",
                "3",
                "--seeds",
                "0,1",
                "--steps",
                "1",
                "--shots",
                "8",
                "--output",
                str(output),
                "--trials-output",
                str(trials),
            ]
        )
        == 0
    )
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    with trials.open() as stream:
        runs = list(csv.DictReader(stream))
    assert len(rows) == 1 and len(runs) == 2
    assert rows[0]["method"] == "F-VQE inverse best feasible"
    assert [row["seed"] for row in runs] == ["0", "1"]
    assert all(row["feasible_rate"] == "1.000000" for row in runs)
    assert output.with_suffix(".problems.json").exists()


def test_budget_configuration_rejects_wrong_type_and_problem():
    from vqe_portfolio.execution import _configuration
    from vqe_portfolio import BinaryVQEConfig

    problem = PortfolioProblem(MU, SIGMA, 2, "binary_qubo", 2)
    cfg = _configuration(problem, "f-vqe", None, 7)
    assert isinstance(cfg, FVQEConfig) and cfg.seed == 7 and cfg.lam == 2
    with pytest.raises(ValueError, match="FVQEConfig"):
        _configuration(problem, "f-vqe", BinaryVQEConfig(), 0)
    with pytest.raises(ValueError, match="match the prepared problem"):
        _configuration(problem, "f-vqe", FVQEConfig(lam=3), 0)
