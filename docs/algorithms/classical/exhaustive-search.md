# Exhaustive binary search

**Track:** exact-K binary selection. **Method ID:** `exact-binary`.

Enumerates every size-K subset and returns the one with the smallest full mean–variance objective.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `k` fixes the subset size; `alpha` is accepted for objective scoring and vanishes for feasible selections.

**Guarantees and limits:** Finds a global binary optimum by enumeration, subject to floating-point scoring. Work grows with the binomial coefficient, so this is a small-instance reference.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import exhaustive_binary_baseline

x, objective = exhaustive_binary_baseline(mu, Sigma, lam=4.0, alpha=0.0, k=2)
```
