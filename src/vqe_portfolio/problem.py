"""Frozen portfolio instances and independent, penalty-free evaluation."""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np


def portfolio_objective(candidate, mu, Sigma, lam) -> float:
    """Original mean-variance objective on raw selections or simplex weights."""
    candidate = np.asarray(candidate, dtype=float)
    return float(lam * candidate @ Sigma @ candidate - mu @ candidate)


def binary_objective(x, mu, Sigma, lam, alpha, k) -> float:
    """Legacy QUBO energy, including the cardinality penalty."""
    x = np.asarray(x, dtype=float)
    return portfolio_objective(x, mu, Sigma, lam) + float(alpha * (x.sum() - k) ** 2)


def fractional_objective(w, mu, Sigma, lam) -> float:
    return portfolio_objective(w, mu, Sigma, lam)


@dataclass(frozen=True, eq=False)
class PortfolioProblem:
    mu: np.ndarray
    Sigma: np.ndarray
    lam: float
    objective_family: str
    k: int | None = None
    problem_id: str = field(init=False)

    def __post_init__(self):
        if np.iscomplexobj(self.mu) or np.iscomplexobj(self.Sigma):
            raise ValueError("Problem data must be real.")
        mu, sigma = np.asarray(self.mu, dtype=float), np.asarray(
            self.Sigma, dtype=float
        )
        if mu.ndim != 1 or not mu.size or sigma.shape != (mu.size, mu.size):
            raise ValueError("Expected nonempty mu and a matching square covariance.")
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
            raise ValueError("Problem data must be finite.")
        if not np.isfinite(self.lam) or self.lam < 0:
            raise ValueError("lam must be finite and nonnegative.")
        if not np.allclose(sigma, sigma.T, atol=1e-12, rtol=1e-10):
            raise ValueError("Covariance must be symmetric.")
        if self.objective_family == "binary_qubo":
            if (
                isinstance(self.k, bool)
                or not isinstance(self.k, (int, np.integer))
                or not 0 <= self.k <= len(mu)
            ):
                raise ValueError("k must be an integer between zero and n_assets.")
            object.__setattr__(self, "k", int(self.k))
        elif self.objective_family != "fractional_simplex" or self.k is not None:
            raise ValueError("Use binary_qubo with k or fractional_simplex without k.")
        # Bytes-backed copies cannot be made writable by an adapter or caller.
        for name, value in [("mu", mu), ("Sigma", (sigma + sigma.T) / 2)]:
            frozen = np.frombuffer(value.astype("<f8").tobytes(), dtype="<f8").reshape(
                value.shape
            )
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "lam", float(self.lam))
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        object.__setattr__(
            self, "problem_id", hashlib.sha256(payload.encode()).hexdigest()
        )

    @property
    def reported_weighting(self):
        return "equal_weight_selected" if self.k is not None else "simplex_weights"

    def to_dict(self):
        """JSON-compatible snapshot; penalty strength is not part of the problem."""
        return {
            "schema_version": 1,
            "mu": self.mu.tolist(),
            "Sigma": self.Sigma.tolist(),
            "lam": self.lam,
            "objective_family": self.objective_family,
            "k": self.k,
            "reported_weighting": self.reported_weighting,
            "simplex_tolerance": 1e-8,
        }

    @classmethod
    def from_dict(cls, snapshot):
        if (
            snapshot.get("schema_version") != 1
            or snapshot.get("simplex_tolerance") != 1e-8
        ):
            raise ValueError("Unsupported problem snapshot.")
        problem = cls(
            snapshot["mu"],
            snapshot["Sigma"],
            snapshot["lam"],
            snapshot["objective_family"],
            snapshot["k"],
        )
        if snapshot.get("reported_weighting") != problem.reported_weighting:
            raise ValueError("Snapshot weighting does not match its objective family.")
        return problem

    def evaluate(self, candidate, *, alpha=None):
        """Validate without projecting/rounding; score feasible portfolios only.

        Binary objectives use x, while risk/return use x/K. Penalty energy is
        diagnostic, never the ranking objective. Invalid candidates retain a
        reason and (when computable) energy but have no scored objective.
        """
        if alpha is not None and (not np.isfinite(alpha) or alpha < 0):
            raise ValueError("alpha must be finite and nonnegative.")
        if candidate is None:
            return PortfolioEvaluation(False, reason="missing_candidate")
        try:
            raw = np.asarray(candidate)
            if np.iscomplexobj(raw):
                return PortfolioEvaluation(False, reason="complex_candidate")
            x = np.asarray(candidate, dtype=float)
        except (TypeError, ValueError):
            return PortfolioEvaluation(False, reason="invalid_candidate")
        if x.shape != self.mu.shape:
            return PortfolioEvaluation(False, reason="wrong_shape")
        if not np.all(np.isfinite(x)):
            return PortfolioEvaluation(False, reason="nonfinite_candidate")
        objective = portfolio_objective(x, self.mu, self.Sigma, self.lam)
        energy = None
        if self.k is not None:
            if alpha is not None:
                energy = binary_objective(
                    x, self.mu, self.Sigma, self.lam, alpha, self.k
                )
            feasible = np.all((x == 0) | (x == 1)) and x.sum() == self.k
            reason = "binary_or_cardinality_violation"
            w = x / self.k if self.k else x
        else:
            feasible = np.all(x >= -1e-8) and abs(x.sum() - 1) <= 1e-8
            reason = "simplex_violation"
            w = x
        if not feasible:
            return PortfolioEvaluation(False, penalized_energy=energy, reason=reason)
        variance = float(w @ self.Sigma @ w)
        ret = float(self.mu @ w)
        if (
            not np.isfinite(objective)
            or not np.isfinite(ret)
            or not np.isfinite(variance)
        ):
            return PortfolioEvaluation(False, reason="nonfinite_metrics")
        if variance < -1e-10:
            return PortfolioEvaluation(False, reason="negative_variance")
        return PortfolioEvaluation(
            True, objective, ret, float(np.sqrt(max(0, variance))), energy
        )


@dataclass(frozen=True)
class PortfolioEvaluation:
    feasible: bool
    objective: float | None = None
    ret: float | None = None
    risk: float | None = None
    penalized_energy: float | None = None
    reason: str = ""
