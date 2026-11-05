"""Reference-free penalty selection and training normalization."""

import numpy as np

from .problem import PortfolioProblem, portfolio_objective


def resolve_penalty(mu, sigma, lam, k, alpha):
    """Resolve a numeric penalty or a conservative sufficient ``'auto'`` value.

    L is a coefficient-wise lower bound on every unpenalized bitstring score;
    U is a feasible top-return score. alpha > U-L excludes all infeasible
    global minimizers since any cardinality violation has squared size >= 1.
    No enumeration, reference optimum, or target information is used.
    """
    if isinstance(alpha, str):
        if alpha != "auto":
            raise ValueError("alpha must be nonnegative or 'auto'")
        problem = PortfolioProblem(mu, sigma, lam, "binary_qubo", k)
        linear = lam * np.diag(problem.Sigma) - problem.mu
        pairs = 2 * lam * problem.Sigma[np.triu_indices(len(mu), 1)]
        lower = np.minimum(linear, 0).sum() + np.minimum(pairs, 0).sum()
        x = np.zeros(len(mu))
        if k:
            x[np.argsort(-problem.mu, kind="stable")[:k]] = 1
        upper = portfolio_objective(x, problem.mu, problem.Sigma, lam)
        span = max(0.0, float(upper - lower))
        alpha = span + max(1e-10, abs(span) * 1e-8)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative or 'auto'")
    return float(alpha)


def energy_scale(hamiltonian):
    """L1 scale excluding identity, which has no effect on the gradient."""
    return max(
        1e-12,
        sum(
            abs(float(c))
            for c, op in zip(hamiltonian.coeffs, hamiltonian.ops)
            if op.name != "Identity"
        ),
    )
