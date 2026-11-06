# Binary MILP (HiGHS)

**Track:** exact-K binary selection. **Method ID:** `highs`.

Linearizes each pairwise binary product using an auxiliary variable and McCormick constraints, then solves the exact-K MILP through SciPy's HiGHS interface.

**Dependencies:** Install `vqe-portfolio[classical]` for SciPy.

**Key parameters:** `time_limit` limits native solver time; `mip_rel_gap` sets its relative-gap target.

**Guarantees and limits:** The formulation is equivalent to the binary quadratic objective and has quadratic model size. Inspect status, incumbent, lower bound, and gap; a time limit does not imply optimality. Model construction is outside the native time limit.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import mixed_integer_binary_baseline

result = mixed_integer_binary_baseline(mu, Sigma, lam=4.0, k=2, time_limit=10.0)
```
