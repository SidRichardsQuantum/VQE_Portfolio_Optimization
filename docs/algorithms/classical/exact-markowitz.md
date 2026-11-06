# Markowitz support enumeration

**Track:** long-only fractional allocation. **Method ID:** `exact-fractional`.

Enumerates nonempty supports, solves each equality-constrained quadratic stationarity system, filters negative weights, and keeps the best feasible allocation. Singular systems use least squares.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `lam` sets risk aversion; `tol` permits small numerical negative weights before clipping and normalization.

**Guarantees and limits:** A small-instance reference for the convex problem with positive semidefinite covariance. It uses numerical linear algebra rather than certified exact arithmetic, and support enumeration grows exponentially. Zero risk aversion selects a maximum-return vertex.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import exact_long_only_markowitz_baseline

weights, objective = exact_long_only_markowitz_baseline(mu, Sigma, lam=4.0)
```
