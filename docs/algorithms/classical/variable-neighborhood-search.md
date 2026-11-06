# Variable neighborhood search (VNS)

**Track:** exact-K binary selection. **Method ID:** `vns`.

Shakes a locally refined portfolio by exchanging q distinct selected and unselected assets, then performs improving single swaps. Strict improvement resets q to one; failure advances q, wrapping after the largest permitted neighborhood.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `seed`, `restarts`, `max_iterations`, `max_neighborhood`, `max_swaps`, and an optional `candidate_callback`.

**Guarantees and limits:** Neighborhood size is capped at min(max_neighborhood, K, n-K). Every candidate preserves K. Returns the best feasible incumbent; the finite search and local refinements do not certify a global optimum.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import variable_neighborhood_search_binary_baseline

result = variable_neighborhood_search_binary_baseline(mu, Sigma, lam=4.0, k=2, seed=0, restarts=2)
```
