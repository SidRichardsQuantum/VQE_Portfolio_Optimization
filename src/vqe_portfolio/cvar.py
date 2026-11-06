"""Lower-tail CVaR of penalized binary portfolio costs.

Analytic CVaR differentiates probabilities, keeping the energy ordering fixed.
Empirical CVaR uses circuit samples and SPSA, never gradients through samples.
"""

from numbers import Real

import numpy as onp
import pennylane as qml
from pennylane import numpy as np

from .optimize import adam_optimize, spsa_optimize
from .reliability import resolve_penalty


def validate_training_objective(cfg):
    if cfg.training_objective not in {"expectation", "cvar"}:
        raise ValueError("training_objective must be 'expectation' or 'cvar'")
    if (
        isinstance(cfg.cvar_alpha, (bool, onp.bool_))
        or not isinstance(cfg.cvar_alpha, Real)
        or not onp.isfinite(cfg.cvar_alpha)
        or not 0 < cfg.cvar_alpha <= 1
    ):
        raise ValueError("cvar_alpha must be a finite number in (0, 1]")
    if cfg.shots_train is not None and (
        type(cfg.shots_train) is not int or cfg.shots_train < 1
    ):
        raise ValueError("shots_train must be a positive integer or None")


def training_metadata(cfg):
    sampled_cvar = (
        cfg.training_objective == "cvar"
        and cfg.cvar_alpha < 1
        and cfg.shots_train is not None
    )
    return {
        "training_objective": cfg.training_objective,
        "cvar_alpha": cfg.cvar_alpha if cfg.training_objective == "cvar" else None,
        "training_optimizer": "spsa" if sampled_cvar else "adam",
    }


def _portfolio_costs(bits, mu, sigma, lam, penalty, k):
    """One classical objective per bitstring, including infeasible samples."""
    return (
        lam * onp.einsum("bi,ij,bj->b", bits, sigma, bits)
        - bits @ mu
        + penalty * (bits.sum(axis=1) - k) ** 2
    )


def _lower_tail_from_sorted(probabilities, energies, fraction):
    """Integrate exactly `fraction` mass, splitting the boundary atom.

    Energies are constant and sorted; only probabilities are differentiated.
    At a quantile boundary CVaR is piecewise differentiable.
    """
    before = np.cumsum(probabilities) - probabilities
    weights = np.minimum(probabilities, np.maximum(fraction - before, 0.0))
    return np.sum(weights * energies) / fraction


def _sample_cvar(costs, fraction):
    """Empirical lower-tail mean with a fractional final observation."""
    ordered = onp.sort(onp.asarray(costs, dtype=float))
    mass = fraction * len(ordered)
    weights = onp.clip(mass - onp.arange(len(ordered)), 0.0, 1.0)
    return float(weights @ ordered / mass)


def make_training_cost(energy, ansatz, device, cfg, mu, sigma, lam):
    """Return the configured cost; expectation and alpha=1 retain legacy QNodes."""
    if cfg.training_objective == "expectation" or cfg.cvar_alpha == 1:
        return energy

    n = len(mu)
    mu, sigma = onp.asarray(mu), onp.asarray(sigma)
    penalty = resolve_penalty(mu, sigma, lam, cfg.k, cfg.alpha)
    if cfg.shots_train is not None:

        @qml.qnode(device)
        def samples(params):
            ansatz(params)
            return qml.sample(wires=range(n))

        samples = qml.set_shots(samples, cfg.shots_train)

        def sampled_cost(params):
            bits = onp.asarray(samples(params)).reshape(cfg.shots_train, n)
            costs = _portfolio_costs(bits, mu, sigma, lam, penalty, cfg.k)
            return _sample_cvar(costs, cfg.cvar_alpha)

        return sampled_cost

    # O(n * 2**n) storage, without forming a dense Hamiltonian matrix.
    indices = onp.arange(2**n)
    bits = (indices[:, None] >> onp.arange(n - 1, -1, -1)) & 1
    energies = _portfolio_costs(bits, mu, sigma, lam, penalty, cfg.k)
    order = onp.argsort(energies, kind="stable")
    sorted_energies = np.array(energies[order], requires_grad=False)

    @qml.qnode(device, interface="autograd")
    def probabilities(params):
        ansatz(params)
        return qml.probs(wires=range(n))

    def analytic_cost(params):
        return _lower_tail_from_sorted(
            probabilities(params)[order], sorted_energies, cfg.cvar_alpha
        )

    return analytic_cost


def optimize_training(cost_fn, init_params, cfg, **kwargs):
    """Choose an optimizer appropriate to the configured cost estimator."""
    if training_metadata(cfg)["training_optimizer"] == "spsa":
        return spsa_optimize(cost_fn, init_params, seed=cfg.seed, **kwargs)
    return adam_optimize(cost_fn, init_params, **kwargs)
