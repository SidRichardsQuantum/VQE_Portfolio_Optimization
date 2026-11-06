# QP relaxation, rounding, and refinement

**Track:** exact-K binary selection. **Method ID:** `qp-round`.

Relaxes selection to 0 <= z <= 1 with sum(z) = K, solves the original quadratic objective using CVXPY/OSQP, selects the largest K entries, and refines with improving swaps. Exact rounding ties follow asset index order.

**Dependencies:** Install `vqe-portfolio[markowitz]` or `vqe-portfolio[classical]`.

**Key parameters:** `qp_max_iter` caps QP iterations; `max_swaps` caps local refinement.

**Guarantees and limits:** Requires positive semidefinite covariance. The relaxation retains quadratic diagonal terms and the raw selection scale. The rounded result is heuristic; numerical relaxation diagnostics are not certified bounds, and failed relaxation output is reported without substituting another solver.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import relaxation_rounding_binary_baseline

result = relaxation_rounding_binary_baseline(mu, Sigma, lam=4.0, k=2)
```
