from dataclasses import replace
from itertools import product

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from vqe_portfolio.binary import build_ising_hamiltonian, run_binary_vqe
from vqe_portfolio.optimize import adam_optimize
from vqe_portfolio.qaoa import _qaoa_ansatz, xy_mixer_edges
from vqe_portfolio.reliability import resolve_penalty
from vqe_portfolio.types import BinaryVQEConfig


@pytest.mark.parametrize("seed", range(5))
def test_auto_penalty_excludes_all_infeasible_ground_states(seed):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=4)
    matrix = rng.normal(size=(4, 4))
    sigma = matrix @ matrix.T
    bits = np.array(list(product((0, 1), repeat=4)))
    f = 3 * np.einsum("bi,ij,bj->b", bits, sigma, bits) - bits @ mu
    for k in range(1, 5):
        alpha = resolve_penalty(mu, sigma, 3.0, k, "auto")
        energy = f + alpha * (bits.sum(axis=1) - k) ** 2
        assert all(
            bits[i].sum() == k
            for i in np.flatnonzero(np.isclose(energy, energy.min(), atol=1e-9, rtol=0))
        )


@pytest.mark.parametrize("topology", ["ring", "complete"])
def test_dicke_xy_preserves_cardinality_and_uses_first_cost_angle(topology):
    mu = np.array([0.1, 0.3, 0.5, 0.2])
    h = build_ising_hamiltonian(mu, np.diag([0.1, 0.4, 0.2, 0.3]), 4, 2, 2)
    device = qml.device("default.qubit", wires=4)

    @qml.qnode(device)
    def probabilities(gamma, initial):
        _qaoa_ansatz(pnp.array([[gamma, 0.23]]), h, 4, 2, "xy", initial, topology)
        return qml.probs(wires=range(4))

    feasible = np.array([i.bit_count() == 2 for i in range(16)])
    uniform = probabilities(0.0, "dicke")
    shifted = probabilities(0.37, "dicke")
    assert shifted[~feasible].sum() == pytest.approx(0, abs=1e-12)
    assert shifted[feasible].sum() == pytest.approx(1)
    assert np.linalg.norm(shifted - uniform) > 1e-3
    np.testing.assert_allclose(
        probabilities(0.0, "basis"), probabilities(0.37, "basis"), atol=1e-12
    )


def test_complete_edges_have_no_duplicates():
    assert len(xy_mixer_edges(5, "complete")) == 10
    assert xy_mixer_edges(2) == ((0, 1),)
    with pytest.raises(ValueError):
        xy_mixer_edges(4, "invalid")


def test_optimizer_logs_matching_post_update_cost_and_keeps_best():
    callbacks = []

    def cost(p):
        return pnp.sum(p**2)

    init = pnp.array([0.05], requires_grad=True)
    result = adam_optimize(
        cost,
        init,
        steps=5,
        stepsize=0.5,
        log_every=2,
        keep_best=True,
        stepsize_decay=0.1,
        callback=lambda step, value, params: callbacks.append(
            (step, value, params.copy())
        ),
    )
    assert result.steps == [2, 4, 5]
    for _, value, params in callbacks:
        assert value == pytest.approx(float(cost(params)))
    assert float(cost(result.params)) <= float(cost(init))


def test_binary_restarts_normalization_and_callback(toy_problem):
    mu, sigma = toy_problem
    cfg = BinaryVQEConfig(
        k=2,
        steps=3,
        restarts=2,
        alpha="auto",
        normalize_energy=True,
        keep_best=True,
        stepsize_decay=0.05,
        log_every=2,
        shots_sample=16,
    )
    events = []
    result = run_binary_vqe(
        mu,
        sigma,
        cfg,
        candidate_callback=lambda x, details: events.append((x, details)),
    )
    assert len(result.restart_energies) == 2
    assert result.effective_alpha == pytest.approx(
        resolve_penalty(mu, sigma, cfg.lam, cfg.k, "auto")
    )
    assert result.energy_scale > 0
    assert len(events) == 5
    assert all(x.sum() == 2 for x, _ in events)
    h = build_ising_hamiltonian(mu, sigma, cfg.lam, result.effective_alpha, cfg.k)

    @qml.qnode(qml.device("default.qubit", wires=3))
    def energy(params):
        from vqe_portfolio.ansatz import apply_binary_ansatz

        apply_binary_ansatz(cfg.ansatz, params, cfg.depth, 3)
        return qml.expval(h)

    assert float(energy(result.params)) == pytest.approx(min(result.restart_energies))
    # Restarts do not use reference information and repeat deterministically.
    repeated = run_binary_vqe(mu, sigma, cfg, candidate_callback=lambda *args: None)
    np.testing.assert_allclose(result.params, repeated.params)
    with pytest.raises(ValueError, match="restarts"):
        run_binary_vqe(mu, sigma, replace(cfg, restarts=0))
