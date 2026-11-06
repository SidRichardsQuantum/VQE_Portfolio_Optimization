# GRASP

**Track:** exact-K binary selection. **Method ID:** `grasp`.

Greedy randomized adaptive search constructs a subset from empty using full-objective marginal insertion costs. At each step it samples from a restricted candidate list, then improves the completed subset with best-improvement swaps.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `seed`, `restarts`, `rcl_alpha`, `max_swaps`, and an optional `candidate_callback`.

**Guarantees and limits:** The candidate threshold is min_cost + rcl_alpha * (max_cost - min_cost). Zero chooses greedy minimum-cost assets with random ties; one allows every remaining asset. Only completed exact-K incumbents are reported, without an optimality bound. `rcl_alpha` is distinct from the quantum cardinality penalty `alpha`.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import grasp_binary_baseline

result = grasp_binary_baseline(mu, Sigma, lam=4.0, k=2, seed=0, restarts=2)
```
