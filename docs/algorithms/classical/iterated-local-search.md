# Iterated local search (ILS)

**Track:** exact-K binary selection. **Method ID:** `ils`.

Refines each start with improving swaps, perturbs the current portfolio by exchanging several distinct assets, then refines again. Only a strictly better refined portfolio replaces the current state.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `seed`, `restarts`, `max_iterations`, `perturbation_size`, `max_swaps`, and an optional `candidate_callback`.

**Guarantees and limits:** Perturbation size is capped at min(K, n-K), preserving cardinality. Returns the best feasible incumbent across restarts; finite perturb-and-refine search has no global optimality guarantee.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import iterated_local_search_binary_baseline

result = iterated_local_search_binary_baseline(mu, Sigma, lam=4.0, k=2, seed=0, restarts=2)
```
