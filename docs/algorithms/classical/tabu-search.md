# Tabu search

**Track:** exact-K binary selection. **Method ID:** `tabu`.

Takes the best admissible swap, including uphill moves. The reverse swap is tabu for a fixed tenure; aspiration permits a tabu move that improves the global incumbent. Memory resets at each restart.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `seed`, `restarts`, `max_iterations`, `tabu_tenure`, and an optional `candidate_callback`.

**Guarantees and limits:** Preserves K and returns the best feasible incumbent. Stops at the iteration cap or when no move is admissible. Neither stopping condition certifies global optimality.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import tabu_search_binary_baseline

result = tabu_search_binary_baseline(mu, Sigma, lam=4.0, k=2, seed=0, restarts=2)
```
