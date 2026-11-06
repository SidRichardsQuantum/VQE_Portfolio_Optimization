# Memetic genetic algorithm

**Track:** exact-K binary selection. **Method ID:** `memetic`.

Combines tournament selection, crossover that keeps shared assets and fills from the parent union, single-swap mutation, and best-improvement local refinement of initial members and offspring. Elitism retains the best member; populations of at least three also receive a random immigrant.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `seed`, `restarts`, `population_size`, `generations`, `mutation_rate`, `tournament_size`, `max_swaps`, and an optional `candidate_callback`.

**Guarantees and limits:** Construction and variation preserve K. Duplicate members are allowed, including when the population exceeds the feasible-space size. Returns the best feasible incumbent without an optimality bound.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import memetic_genetic_binary_baseline

result = memetic_genetic_binary_baseline(mu, Sigma, lam=4.0, k=2, seed=0, restarts=2)
```
