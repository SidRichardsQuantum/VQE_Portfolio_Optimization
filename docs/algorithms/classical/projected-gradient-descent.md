# Projected gradient descent (PGD)

**Track:** long-only fractional allocation. **Method ID:** `pgd`.

Starts at equal weights, steps along the negative quadratic gradient, and projects onto the probability simplex. Uses the covariance spectrum for an initial step size and backtracks to enforce quadratic majorization and numerical non-increase.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `max_iterations`, `tolerance`, `stepsize`, and an optional `candidate_callback`.

**Guarantees and limits:** Requires positive semidefinite covariance within the implementation's eigenvalue tolerance. Stops on a numerical first-order gap, grad(f) @ w - min(grad(f)). Zero curvature uses a linear simplex solution. Check convergence or limit status; the reported gap is not independently certified.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import projected_gradient_descent_baseline

result = projected_gradient_descent_baseline(mu, Sigma, lam=4.0)
```
