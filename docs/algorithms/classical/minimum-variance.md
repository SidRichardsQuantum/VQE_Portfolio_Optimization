# Minimum-variance binary selection

**Track:** exact-K binary selection. **Interface:** Python baseline API.

Enumerates all size-K subsets, selects the one with minimum variance, and reports its full mean–variance objective.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `k` fixes the subset size; `lam` and `alpha` affect reported scoring.

**Guarantees and limits:** Globally minimizes variance over the enumerated subsets, subject to floating-point scoring. It does not optimize the return term; enumeration is practical only for small instances.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import minimum_variance_binary_baseline

x, objective = minimum_variance_binary_baseline(mu, Sigma, lam=4.0, alpha=0.0, k=2)
```
