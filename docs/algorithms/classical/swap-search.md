# Multistart swap local search

**Track:** exact-K binary selection. **Method ID:** `swap`.

Starts from top returns, diagonal objective costs, and seeded random subsets. Repeatedly applies the best improving exchange of one selected and one unselected asset.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `seed`, `restarts` (at least two), `max_swaps`, and an optional `candidate_callback`.

**Guarantees and limits:** Every exchange preserves K. An uncapped descent reaches a one-swap local optimum within numerical tolerance; the swap cap can stop earlier. Neither implies a global optimum.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import local_search_binary_baseline

result = local_search_binary_baseline(mu, Sigma, lam=4.0, k=2, seed=0, restarts=2)
```
