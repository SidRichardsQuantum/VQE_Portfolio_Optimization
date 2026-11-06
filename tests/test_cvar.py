"""CVaR mathematics, gradients, solver integration and CLI regression checks."""

from dataclasses import replace
import json

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import pytest

from vqe_portfolio import (
    BinaryVQEConfig,
    QAOAConfig,
    LambdaSweepConfig,
    binary_lambda_sweep,
    qaoa_lambda_sweep,
    run_binary_vqe,
    run_qaoa,
)
from vqe_portfolio.ansatz import apply_binary_ansatz
from vqe_portfolio.binary import build_ising_hamiltonian
from vqe_portfolio.cli import main
from vqe_portfolio.cvar import (
    _lower_tail_from_sorted,
    _sample_cvar,
    make_training_cost,
)
from vqe_portfolio.optimize import spsa_optimize
from vqe_portfolio.qaoa import _qaoa_ansatz


MU = np.array([0.1, 0.2, 0.15])
SIGMA = np.array([[0.05, 0.01, 0], [0.01, 0.06, 0.01], [0, 0.01, 0.04]])


@pytest.mark.parametrize("fraction,expected", [(0.1, -4), (0.25, -2.2), (1, 4.15)])
def test_analytic_tail_splits_probability_at_quantile(fraction, expected):
    assert _lower_tail_from_sorted(
        pnp.array([0.1, 0.25, 0.15, 0.5]), pnp.array([-4, -1, 2, 9]), fraction
    ) == pytest.approx(expected)


@pytest.mark.parametrize("fraction,expected", [(0.1, -4), (0.375, -3), (1, 1.5)])
def test_empirical_tail_uses_fractional_boundary_sample(fraction, expected):
    assert _sample_cvar([9, -1, 2, -4], fraction) == pytest.approx(expected)


def test_tail_handles_ties_zero_mass_and_tiny_fractions():
    assert _lower_tail_from_sorted(
        pnp.array([0, 0.1, 0.4, 0.5]), pnp.array([-100, -2, -2, 5]), 0.25
    ) == pytest.approx(-2)
    assert _sample_cvar([2, 2, 5], 1e-12) == pytest.approx(2)


@pytest.mark.parametrize("penalty", [2.0, "auto"])
def test_analytic_cvar_gradient_matches_finite_differences(penalty):
    cfg = BinaryVQEConfig(
        k=2, lam=1, alpha=penalty, training_objective="cvar", cvar_alpha=0.73
    )
    dev = qml.device("default.qubit", wires=3)

    def ansatz(theta):
        for i in range(3):
            qml.RY(theta[i], wires=i)
        qml.CNOT(wires=[0, 1])

    cost = make_training_cost(None, ansatz, dev, cfg, MU, SIGMA, cfg.lam)
    theta = pnp.array([0.7, 1.1, 0.9], requires_grad=True)
    gradient = qml.grad(cost)(theta)
    finite = []
    for direction in np.eye(3) * 1e-6:
        finite.append(
            (float(cost(theta + direction)) - float(cost(theta - direction))) / 2e-6
        )
    np.testing.assert_allclose(gradient, finite, atol=1e-8)
    assert np.linalg.norm(gradient) > 1e-3


SOLVERS = [(run_binary_vqe, BinaryVQEConfig), (run_qaoa, QAOAConfig)]


@pytest.mark.parametrize("solver,config_type", SOLVERS)
@pytest.mark.parametrize("shots", [None, 32])
def test_full_tail_reproduces_existing_mean_training(solver, config_type, shots):
    cfg = config_type(
        k=2,
        steps=3,
        stepsize=0.02,
        log_every=1,
        shots_train=shots,
        shots_sample=32,
        seed=17,
    )
    mean = solver(MU, SIGMA, cfg)
    full_tail = solver(MU, SIGMA, replace(cfg, training_objective="cvar", cvar_alpha=1))
    np.testing.assert_allclose(mean.params, full_tail.params, atol=1e-12)
    mean_trace = (
        mean.energy_trace if config_type is BinaryVQEConfig else mean.cost_trace
    )
    tail_trace = (
        full_tail.energy_trace
        if config_type is BinaryVQEConfig
        else full_tail.cost_trace
    )
    np.testing.assert_allclose(mean_trace.values, tail_trace.values, atol=1e-12)
    assert mean.sample_counts == full_tail.sample_counts
    assert full_tail.training_optimizer == "adam"
    assert full_tail.training_objective == "cvar"
    assert full_tail.cvar_alpha == 1


def _independent_cost(result, cfg):
    h = build_ising_hamiltonian(MU, SIGMA, cfg.lam, cfg.alpha, cfg.k)

    @qml.qnode(qml.device("default.qubit", wires=3))
    def probabilities():
        if isinstance(cfg, BinaryVQEConfig):
            apply_binary_ansatz(cfg.ansatz, result.params, cfg.depth, 3)
        else:
            _qaoa_ansatz(
                result.params,
                h,
                3,
                cfg.k,
                cfg.mixer,
                cfg.xy_initial_state,
                cfg.xy_topology,
            )
        return qml.probs(wires=range(3))

    # Independent diagonal Hamiltonian reference, with explicit tail integration.
    costs = np.diag(qml.matrix(h, wire_order=range(3))).real
    distribution = np.asarray(probabilities())
    remaining, total = cfg.cvar_alpha, 0.0
    for index in np.argsort(costs):
        mass = min(remaining, distribution[index])
        total += mass * costs[index]
        remaining -= mass
        if remaining <= 0:
            break
    return total / cfg.cvar_alpha


@pytest.mark.parametrize("solver,config_type", SOLVERS)
def test_training_trace_is_cvar_in_original_units_and_callbacks_remain_feasible(
    solver, config_type
):
    options = dict(normalize_energy=True) if config_type is BinaryVQEConfig else {}
    cfg = config_type(
        k=2,
        lam=1,
        alpha="auto",
        steps=4,
        stepsize=0.02,
        log_every=3,
        shots_sample=64,
        training_objective="cvar",
        cvar_alpha=0.8,
        **options,
    )
    candidates = []
    result = solver(
        MU, SIGMA, cfg, candidate_callback=lambda x, _: candidates.append(x)
    )
    trace = result.energy_trace if config_type is BinaryVQEConfig else result.cost_trace
    assert trace.steps == [3, 4]
    assert trace.values[-1] == pytest.approx(_independent_cost(result, cfg), abs=1e-10)
    assert result.training_optimizer == "adam"
    assert candidates and all(np.sum(x) == cfg.k for x in candidates)
    assert sum(result.sample_counts.values()) == cfg.shots_sample


def test_restarts_and_best_iterate_select_by_cvar():
    cfg = BinaryVQEConfig(
        k=2,
        lam=1,
        steps=3,
        stepsize=0.02,
        restarts=3,
        normalize_energy=True,
        keep_best=True,
        training_objective="cvar",
        cvar_alpha=0.8,
        shots_sample=16,
    )
    result = run_binary_vqe(MU, SIGMA, cfg)
    assert _independent_cost(result, cfg) == pytest.approx(min(result.restart_energies))
    assert len(result.restart_energies) == 3


@pytest.mark.parametrize("solver,config_type", SOLVERS)
@pytest.mark.parametrize("shots", [1, 64])
def test_sampled_cvar_uses_spsa_and_repeats_with_seed(solver, config_type, shots):
    cfg = config_type(
        k=2,
        lam=1,
        steps=3,
        log_every=2,
        shots_sample=16,
        shots_train=shots,
        training_objective="cvar",
        cvar_alpha=0.375,
        stepsize=0.02,
    )
    result = solver(MU, SIGMA, cfg)
    repeated = solver(MU, SIGMA, cfg)
    np.testing.assert_allclose(result.params, repeated.params)
    assert result.sample_counts == repeated.sample_counts
    assert result.training_optimizer == "spsa"
    trace = result.energy_trace if config_type is BinaryVQEConfig else result.cost_trace
    assert trace.steps == [2, 3]
    assert np.all(np.isfinite(trace.values))


def test_sampled_cost_scores_all_samples_without_enumeration_or_probabilities(
    monkeypatch,
):
    import vqe_portfolio.cvar as cvar

    # This known state is infeasible: dropping samples would hide its penalty.
    cfg = BinaryVQEConfig(k=2, lam=1, shots_train=7, training_objective="cvar")

    def ansatz(_):
        qml.PauliX(wires=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Finite-shot CVaR must not enumerate state probabilities")

    original = cvar._portfolio_costs
    rows_seen = []

    def record(bits, *args):
        rows_seen.append(len(bits))
        return original(bits, *args)

    monkeypatch.setattr(cvar, "_portfolio_costs", record)
    # The simulator itself uses probabilities internally to draw samples.
    # Forbid an application-level probability table during cost construction.
    with monkeypatch.context() as patch:
        patch.setattr(qml, "probs", forbidden)
        cost = make_training_cost(
            None, ansatz, qml.device("default.qubit", wires=3), cfg, MU, SIGMA, cfg.lam
        )
    assert cost(np.array([0])) == pytest.approx(0.05 - 0.1 + 2.0)
    assert rows_seen == [7]


@pytest.mark.parametrize("solver,config_type", SOLVERS)
@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, np.nan, np.inf, True, "0.25"])
def test_reject_invalid_tail_fractions(solver, config_type, fraction):
    cfg = config_type(k=2, steps=0, training_objective="cvar", cvar_alpha=fraction)
    with pytest.raises(ValueError, match="cvar_alpha"):
        solver(MU, SIGMA, cfg)


@pytest.mark.parametrize("solver,config_type", SOLVERS)
def test_reject_unknown_objective_and_invalid_shots(solver, config_type):
    with pytest.raises(ValueError, match="training_objective"):
        solver(MU, SIGMA, config_type(training_objective="invalid"))
    for shots in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="shots_train"):
            solver(MU, SIGMA, config_type(shots_train=shots))


@pytest.mark.parametrize(
    "sweep,config_type",
    [(binary_lambda_sweep, BinaryVQEConfig), (qaoa_lambda_sweep, QAOAConfig)],
)
@pytest.mark.parametrize("shots", [None, 16])
def test_lambda_sweeps_use_cvar_and_preserve_metadata(sweep, config_type, shots):
    cfg = config_type(
        k=2,
        lam=1,
        steps=1,
        shots_sample=16,
        shots_train=shots,
        training_objective="cvar",
        cvar_alpha=0.8,
    )
    result = sweep(
        MU,
        SIGMA,
        cfg,
        LambdaSweepConfig([0.5, 2], steps_per_lambda=2, stepsize=0.02, warm_start=True),
    )
    assert result.probs_by_lambda.shape == (2, 3)
    assert result.training_objective == "cvar"
    assert result.cvar_alpha == cfg.cvar_alpha
    assert result.training_optimizer == ("adam" if shots is None else "spsa")
    with pytest.raises(ValueError, match="cvar_alpha"):
        sweep(MU, SIGMA, replace(cfg, cvar_alpha=0), LambdaSweepConfig([1]))


def test_spsa_improves_smooth_cost_and_logs_matching_snapshots():
    events = []
    initial = np.array([1.0])
    result = spsa_optimize(
        lambda p: np.sum(p**2),
        initial,
        steps=12,
        stepsize=0.2,
        log_every=5,
        keep_best=True,
        callback=lambda t, v, p: events.append((t, v, p)),
    )
    assert result.steps == [5, 10, 12]
    assert np.sum(result.params**2) < np.sum(initial**2)
    assert initial[0] == 1
    for _, value, params in events:
        assert value == pytest.approx(np.sum(params**2))
    zero = spsa_optimize(lambda p: np.sum(p**2), initial, steps=0, stepsize=0.2)
    np.testing.assert_array_equal(zero.params, initial)
    assert zero.values == []
    with pytest.raises(ValueError, match="nonfinite"):
        spsa_optimize(lambda p: np.nan, initial, steps=1, stepsize=0.2)


@pytest.mark.parametrize("command", ["binary", "qaoa"])
@pytest.mark.parametrize("shots", [None, 16])
def test_cli_cvar_outputs_configuration_and_optimizer_metadata(command, shots, capsys):
    args = [
        command,
        "--mu",
        "0.1,0.2",
        "--sigma",
        "0.1,0;0,0.2",
        "--k",
        "1",
        "--lam",
        "1",
        "--steps",
        "2",
        "--shots-sample",
        "16",
        "--training-objective",
        "cvar",
        "--cvar-alpha",
        "0.375",
    ]
    if shots is not None:
        args += ["--shots-train", str(shots)]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["training_objective"] == "cvar"
    assert payload["result"]["cvar_alpha"] == 0.375
    assert payload["result"]["training_optimizer"] == (
        "adam" if shots is None else "spsa"
    )


@pytest.mark.parametrize("mixer", ["x", "xy"])
def test_cvar_qaoa_supports_both_mixers(mixer):
    cfg = QAOAConfig(
        k=2,
        p=1,
        steps=2,
        shots_sample=64,
        mixer=mixer,
        xy_initial_state="dicke" if mixer == "xy" else "basis",
        training_objective="cvar",
        cvar_alpha=0.8,
    )
    result = run_qaoa(MU, SIGMA, cfg)
    assert result.cost_trace.values[-1] == pytest.approx(_independent_cost(result, cfg))
    if mixer == "xy":
        assert all(bitstring.count("1") == cfg.k for bitstring in result.sample_counts)
