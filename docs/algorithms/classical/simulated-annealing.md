# Simulated annealing

**Track:** exact-K binary selection. **Method ID:** `anneal`.

Proposes random one-for-one swaps. Improving moves are accepted; uphill moves are accepted using the Metropolis probability exp(-delta/T). Temperature decreases geometrically and resets at each restart.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `seed`, `restarts`, `max_iterations`, `initial_temperature`, `cooling_rate`, and an optional `candidate_callback`.

**Guarantees and limits:** Maintains exact K and returns the best feasible state across restarts. The default initial temperature uses the largest absolute starting swap cost. Finite geometric cooling and iteration limits provide no global optimality guarantee.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import simulated_annealing_binary_baseline

result = simulated_annealing_binary_baseline(mu, Sigma, lam=4.0, k=2, seed=0, restarts=2)
```
