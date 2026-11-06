# Top-return selection

**Track:** exact-K binary selection. **Method ID:** `top-return`.

Selects the K assets with the largest expected returns, then scores the resulting portfolio with the full objective.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `k` controls cardinality, including the empty selection at zero.

**Guarantees and limits:** Ignores covariance during selection. It is a deterministic comparison baseline, without a general mean–variance optimality guarantee.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import top_return_baseline

x, objective = top_return_baseline(mu, Sigma, lam=4.0, alpha=0.0, k=2)
```
