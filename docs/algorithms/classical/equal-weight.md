# Equal-weight allocation

**Track:** long-only fractional allocation. **Interface:** Python baseline API.

Assigns weight 1/n to every asset and evaluates the fractional objective.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `lam` affects scoring; there are no search parameters.

**Guarantees and limits:** Always produces a simplex allocation for nonempty inputs, but performs no optimization and provides no general optimality guarantee.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import equal_weight_baseline

weights, objective = equal_weight_baseline(mu, Sigma, lam=4.0)
```
