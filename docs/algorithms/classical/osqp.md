# Convex Markowitz QP (OSQP)

**Track:** long-only fractional allocation. **Method ID:** `osqp`.

Builds the long-only Markowitz quadratic program in CVXPY and solves it with OSQP. Small negative numerical weights are clipped, weights are normalized, and the objective is recomputed.

**Dependencies:** Install `vqe-portfolio[markowitz]` or `vqe-portfolio[classical]` for CVXPY and OSQP.

**Key parameters:** `lam` sets risk aversion. The wrapper fixes solver tolerances at 1e-9 and the iteration cap at 100000.

**Guarantees and limits:** Requires positive semidefinite covariance. Inspect the native status and candidate; numerical solver success is not an independently certified lower bound.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import convex_markowitz_baseline

result = convex_markowitz_baseline(mu, Sigma, lam=4.0)
```
