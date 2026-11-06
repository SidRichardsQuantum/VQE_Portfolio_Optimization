# Random exact-K sampling

**Track:** exact-K binary selection. **Method ID:** `random-k`.

Draws uniformly random subsets of K distinct assets, scores them with the full
objective, and retains the best feasible candidate. Sampling is with replacement
across draws, so subsets may repeat. This baseline performs no local refinement.

**Dependencies:** NumPy; no extra dependencies.

**Key parameters:** `seed` and the sampling allowance. Small-instance validation
uses `ValidationConfig.shots` draws; the budget worker samples until interrupted
or stopped by its coverage logic.

**Guarantees and limits:** Every draw preserves K, but a finite random sample
does not certify optimality unless all feasible subsets have actually been
covered. Seeded results also depend on how many draws complete.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples). This is a script
example: keep the main guard because the budget interface spawns a worker.

```python
from vqe_portfolio import PortfolioProblem, run_budgeted

if __name__ == "__main__":
    problem = PortfolioProblem(mu, Sigma, 4.0, "binary_qubo", 2)
    result = run_budgeted(problem, "random-k", budget_seconds=10.0, seed=0)
```

Startup consumes the budget; `result.candidate` can be absent if no feasible
candidate arrives in time.
