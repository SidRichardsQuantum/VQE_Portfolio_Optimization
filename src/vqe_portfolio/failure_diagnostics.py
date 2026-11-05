"""Exact XY support analysis and small binary depth/penalty diagnostics."""

from collections import defaultdict
from dataclasses import replace

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .ansatz import apply_binary_ansatz
from .binary import build_ising_hamiltonian
from .problem import PortfolioProblem
from .qaoa import _qaoa_ansatz, xy_mixer_edges
from .training_diagnostics import target_basis
from .training_effort import measure_trajectory
from .validation import objective_scale


def xy_support_polynomials(n, k):
    """Exact depth-1 mixer amplitudes as integer Laurent polynomials in z.

    z=exp(2j*beta). Each exchange maps |01> to c|01>-i*s|10>,
    with c=(z+z^-1)/2 and -i*s=(z^-1-z)/2. Equal bits are unchanged.
    All amplitudes share denominator 2**len(edges). Integer cancellation is
    exact; a nonzero Laurent polynomial cannot vanish for every real beta.
    The preceding diagonal cost layer adds only a global phase to the basis
    initial state, so this is the exact support over all gamma/beta at p=1.
    """
    if not 1 <= k < n or not 2 <= n <= 16:
        raise ValueError("Require 2–16 wires and 0 < k < n.")
    initial = tuple([1] * k + [0] * (n - k))
    polynomials = {initial: {0: 1}}
    for i, j in xy_mixer_edges(n):
        updated = defaultdict(lambda: defaultdict(int))
        for bits, terms in polynomials.items():
            if bits[i] == bits[j]:
                for power, coefficient in terms.items():
                    updated[bits][power] += 2 * coefficient
            else:
                swapped = list(bits)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                swapped = tuple(swapped)
                for power, coefficient in terms.items():
                    updated[bits][power - 1] += coefficient
                    updated[bits][power + 1] += coefficient
                    updated[swapped][power - 1] += coefficient
                    updated[swapped][power + 1] -= coefficient
        polynomials = {
            bits: {power: value for power, value in terms.items() if value}
            for bits, terms in updated.items()
            if any(terms.values())
        }
    return {
        "".join(map(str, bits)): {
            str(power): value for power, value in sorted(terms.items())
        }
        for bits, terms in sorted(polynomials.items())
    }


def polynomial_probabilities(n, polynomials, beta):
    probabilities = np.zeros(2**n)
    z = np.exp(2j * beta)
    denominator = 2 ** len(xy_mixer_edges(n))
    for bits, terms in polynomials.items():
        amplitude = (
            sum(value * z ** int(power) for power, value in terms.items()) / denominator
        )
        probabilities[int(bits, 2)] = abs(amplitude) ** 2
    return probabilities


def xy_support_upper_bound(n, k, depth):
    """Gate-path support superset, allowing a stay/swap at each unequal pair.

    Cost layers are diagonal and cannot create new basis support. At p>1 this
    bound ignores interference and tied angles; membership alone is no proof
    that a state is reachable. Exclusion certifies unreachability.
    """
    if depth < 1:
        raise ValueError("depth must be positive.")
    support = {tuple([1] * k + [0] * (n - k))}
    for _ in range(depth):
        for i, j in xy_mixer_edges(n):
            updated = set(support)
            for bits in support:
                swapped = list(bits)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                updated.add(tuple(swapped))
            support = updated
    return sorted("".join(map(str, bits)) for bits in support)


def circuit_probabilities(problem, method, params, depth, alpha):
    device = qml.device("default.qubit", wires=len(problem.mu))
    hamiltonian = build_ising_hamiltonian(
        problem.mu, problem.Sigma, problem.lam, alpha, problem.k
    )

    @qml.qnode(device)
    def circuit(theta):
        if method == "qaoa-xy":
            _qaoa_ansatz(
                theta, H_cost=hamiltonian, n=len(problem.mu), k=problem.k, mixer="xy"
            )
        elif method == "binary-vqe":
            apply_binary_ansatz("ry_cz", theta, depth, len(problem.mu))
        else:
            raise ValueError("Unsupported diagnostic method.")
        return qml.probs(wires=range(len(problem.mu)))

    return np.asarray(circuit(np.asarray(params, dtype=float)), dtype=float)


def penalty_diagnostic(problem, reference_objective, penalties):
    bits, feasible, _, objectives = target_basis(problem, reference_objective, 0.001)
    violation = (bits.sum(axis=1) - problem.k) ** 2
    critical = max(
        0.0,
        float(
            np.max((reference_objective - objectives[~feasible]) / violation[~feasible])
        ),
    )
    rows = []
    for alpha in penalties:
        energy = objectives + alpha * violation
        minimum = float(energy.min())
        minimisers = np.isclose(
            energy, minimum, rtol=0, atol=1e-10 * objective_scale(problem)
        )
        rows.append(
            {
                "alpha": alpha,
                "critical_alpha": critical,
                "strict_sufficient": bool(alpha > critical),
                "penalized_minimum": minimum,
                "all_numerical_minimisers_feasible": bool(np.all(feasible[minimisers])),
                "feasible_minimum": reference_objective,
            }
        )
    return rows


def swap_path_barrier(problem, x, neighbour, alpha):
    """Maximum rise along a particular two-rotation product-state swap path.

    This illustrates one path, not a lower bound on all ansatz paths or a
    certificate of an optimiser local minimum.
    """
    removed = np.flatnonzero((x == 1) & (neighbour == 0))
    added = np.flatnonzero((x == 0) & (neighbour == 1))
    if len(removed) != 1 or len(added) != 1:
        raise ValueError("Require one feasible one-for-one swap.")
    minus, plus = x.copy(), x.copy()
    minus[removed[0]] = 0
    plus[added[0]] = 1

    def score(candidate):
        return float(
            problem.lam * candidate @ problem.Sigma @ candidate - problem.mu @ candidate
        )

    f0, f1, fm, fp = map(score, (x, neighbour, minus, plus))
    linear = -2 * f0 + fm + fp + 2 * alpha
    quadratic = f0 + f1 - fm - fp - 2 * alpha
    points = [0.0, 1.0]
    if quadratic < 0 and 0 < -linear / (2 * quadratic) < 1:
        points.append(-linear / (2 * quadratic))
    barrier = max(linear * u + quadratic * u * u for u in points)
    return {
        "barrier": max(0.0, float(barrier)),
        "linear": linear,
        "quadratic": quadratic,
        "initial_objective": f0,
        "neighbour_objective": f1,
    }


def vqe_mode_diagnostic(
    problem, params, probabilities, depth, alpha, reference_objective
):
    mode = np.array(
        list(np.binary_repr(int(np.argmax(probabilities)), width=len(problem.mu))),
        dtype=int,
    )
    evaluation = problem.evaluate(mode, alpha=alpha)
    result = {
        "mode": "".join(map(str, mode)),
        "mode_probability": float(max(probabilities)),
        "mode_feasible": bool(evaluation.feasible),
        "mode_objective": evaluation.objective,
        "mode_normalised_gap": (
            (evaluation.objective - reference_objective) / objective_scale(problem)
            if evaluation.feasible
            else None
        ),
        "best_swap": None,
        "swap_improvement": None,
        "product_swap_barrier": None,
    }
    if evaluation.feasible:
        neighbours = []
        for i in np.flatnonzero(mode):
            for j in np.flatnonzero(1 - mode):
                neighbour = mode.copy()
                neighbour[i], neighbour[j] = 0, 1
                neighbours.append((problem.evaluate(neighbour).objective, neighbour))
        value, neighbour = min(neighbours, key=lambda pair: pair[0])
        result["best_swap"] = "".join(map(str, neighbour))
        result["swap_improvement"] = float(evaluation.objective - value)
        if value < evaluation.objective:
            result["product_swap_barrier"] = swap_path_barrier(
                problem, mode, neighbour, alpha
            )
    hamiltonian = build_ising_hamiltonian(
        problem.mu, problem.Sigma, problem.lam, alpha, problem.k
    )
    device = qml.device("default.qubit", wires=len(problem.mu))

    @qml.qnode(device, interface="autograd")
    def energy(theta):
        apply_binary_ansatz("ry_cz", theta, depth, len(problem.mu))
        return qml.expval(hamiltonian)

    gradient = qml.grad(energy)(pnp.array(params, requires_grad=True))
    result["final_gradient_norm"] = float(np.linalg.norm(gradient))
    return result


def variant_plan(base_config, instances, xy_depths, vqe_depths, penalties):
    for instance in instances:
        for depth in xy_depths:
            for seed in base_config.solver_seeds:
                variant = f"xy-p{depth}"
                key = f"{instance['problem_id']}:{variant}:{seed}"
                yield key, instance, variant, "qaoa-xy", seed, replace(
                    base_config, qaoa_p=depth
                )
        for depth in vqe_depths:
            for alpha in penalties:
                for seed in base_config.solver_seeds:
                    variant = f"vqe-d{depth}-a{alpha:g}"
                    key = f"{instance['problem_id']}:{variant}:{seed}"
                    yield key, instance, variant, "binary-vqe", seed, replace(
                        base_config, binary_depth=depth, alpha=alpha
                    )


def measure_variant(
    key, instance, variant, method, seed, config, steps, draws, baseline=None
):
    from copy import deepcopy

    from .problem import PortfolioProblem

    if baseline is None:
        row = measure_trajectory(key, instance, method, seed, config, steps, draws)
    else:
        row = deepcopy(baseline)
        row["baseline_source_trial_id"] = row["trial_id"]
        row["trial_id"] = key
    row.update(variant=variant, reused_baseline=baseline is not None)
    try:
        if row["error"]:
            return row
        problem = PortfolioProblem.from_dict(instance["snapshot"])
        depth = config.qaoa_p if method == "qaoa-xy" else config.binary_depth
        if baseline is not None:
            for state in row["checkpoints"]:
                actual = circuit_probabilities(
                    problem, method, state["params"], depth, config.alpha
                )
                if not np.allclose(
                    actual, state["probabilities"], rtol=1e-10, atol=1e-12
                ):
                    raise ValueError(
                        "Cached baseline disagrees with the production circuit."
                    )
        if method == "binary-vqe":
            state = row["checkpoints"][-1]
            row["final_vqe_diagnostic"] = vqe_mode_diagnostic(
                problem,
                state["params"],
                state["probabilities"],
                depth,
                config.alpha,
                instance["reference_objective"],
            )
    except Exception as exc:  # noqa: BLE001 - preserve failed variants
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def static_diagnostics(
    instances, xy_depths, vqe_depths, penalties, target_gap, beta_points=257
):
    models, problems, support_rows, beta_rows, parameter_rows, penalty_rows = (
        {},
        {},
        [],
        [],
        [],
        [],
    )
    for instance in instances:
        problem = PortfolioProblem.from_dict(instance["snapshot"])
        if problem.problem_id != instance["problem_id"]:
            raise ValueError("Problem snapshot hash mismatch.")
        n, k = len(problem.mu), problem.k
        bits, feasible, target, _ = target_basis(
            problem, instance["reference_objective"], target_gap
        )
        polynomials = xy_support_polynomials(n, k)
        model_key = f"{n}:{k}"
        models[model_key] = {
            "n_assets": n,
            "k": k,
            "edges": list(xy_mixer_edges(n)),
            "denominator": 2 ** len(xy_mixer_edges(n)),
            "amplitude_laurent_coefficients": polynomials,
        }
        targets = ["".join(map(str, x)) for x in bits[target]]
        info = {
            "instance_seed": instance["instance_seed"],
            "n_assets": n,
            "k": k,
            "target_bitstrings": targets,
            "feasible_bitstrings": ["".join(map(str, x)) for x in bits[feasible]],
            "model_key": model_key,
            "vqe_target_constructions": [],
        }
        for depth in xy_depths:
            support = (
                sorted(polynomials)
                if depth == 1
                else xy_support_upper_bound(n, k, depth)
            )
            support_rows.append(
                {
                    "problem_id": problem.problem_id,
                    "instance_seed": instance["instance_seed"],
                    "depth": depth,
                    "support_kind": (
                        "exact_laurent" if depth == 1 else "gate_path_upper_bound"
                    ),
                    "support_count": len(support),
                    "target_count": len(targets),
                    "covered_targets": len(set(targets) & set(support)),
                    "all_targets_certified_unreachable": not bool(
                        set(targets) & set(support)
                    ),
                }
            )
            info[f"xy_support_{depth}"] = support
            params = np.random.default_rng(314 + depth).uniform(0.2, 1.1, (depth, 2))
            initial = circuit_probabilities(problem, "qaoa-xy", params, depth, 5.0)
            for layer in range(depth):
                for column, name in enumerate(("gamma", "beta")):
                    perturbed = params.copy()
                    perturbed[layer, column] += 0.1
                    actual = circuit_probabilities(
                        problem, "qaoa-xy", perturbed, depth, 5.0
                    )
                    parameter_rows.append(
                        {
                            "problem_id": problem.problem_id,
                            "instance_seed": instance["instance_seed"],
                            "depth": depth,
                            "layer": layer,
                            "parameter": name,
                            "shift": 0.1,
                            "total_variation_change": float(
                                0.5 * np.sum(abs(actual - initial))
                            ),
                        }
                    )
        for beta in np.linspace(0, np.pi, beta_points):
            probabilities = polynomial_probabilities(n, polynomials, beta)
            beta_rows.append(
                {
                    "problem_id": problem.problem_id,
                    "instance_seed": instance["instance_seed"],
                    "beta": float(beta),
                    "p_target": float(probabilities[target].sum()),
                    "p_feasible": float(probabilities[feasible].sum()),
                }
            )
        for row in penalty_diagnostic(
            problem, instance["reference_objective"], penalties
        ):
            penalty_rows.append(
                {
                    "problem_id": problem.problem_id,
                    "instance_seed": instance["instance_seed"],
                    **row,
                }
            )
        for depth in vqe_depths:
            params = np.zeros((depth, n))
            params[0] = np.pi * np.array(instance["reference_candidate"])
            probabilities = circuit_probabilities(
                problem, "binary-vqe", params, depth, 5.0
            )
            info["vqe_target_constructions"].append(
                {
                    "depth": depth,
                    "params": params.tolist(),
                    "p_target": float(probabilities[target].sum()),
                }
            )
        problems[problem.problem_id] = info
    return {
        "models": models,
        "problems": problems,
        "support_rows": support_rows,
        "beta_rows": beta_rows,
        "parameter_rows": parameter_rows,
        "penalty_rows": penalty_rows,
    }
