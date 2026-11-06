# Binary quadratic optimization (SCIP)

**Track:** exact-K binary selection. **Method ID:** `scip`.

Uses binary selection variables, an exact-K constraint, and an unrestricted epigraph variable constrained above the quadratic objective. SCIP solves this formulation without explicit pairwise product variables.

**Dependencies:** Install `vqe-portfolio[scip]` or `vqe-portfolio[classical]` for PySCIPOpt.

**Key parameters:** `time_limit`, `mip_rel_gap`, and an optional `candidate_callback` control solving and incumbent reporting.

**Guarantees and limits:** Retains native status, finite dual-bound diagnostics, and gap. Limits can leave a feasible incumbent without optimality. Model construction and cleanup are outside the native time limit; signed couplings and negative objectives are supported.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import scip_binary_baseline

result = scip_binary_baseline(mu, Sigma, lam=4.0, k=2, time_limit=10.0)
```
