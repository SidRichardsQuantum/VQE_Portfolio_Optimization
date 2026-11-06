# QAOA with X mixing

**Track:** binary asset selection. **Method ID:** `qaoa-x`.

Alternates diagonal cost-phase layers with single-qubit X mixing, starting from a uniform superposition of all bitstrings. Adam optimizes the p pairs of cost and mixer angles using the penalized Ising objective.

Optional [CVaR training](cvar.md) minimizes a lower tail of penalized costs;
finite-shot CVaR uses SPSA instead of Adam.

**Dependencies and execution:** PennyLane and the package's core dependencies.
The default device is the `default.qubit` simulator. Analytic training is the
default; finite-shot training can be requested. Device selection alone does not
demonstrate hardware execution or quantum advantage.

**Key parameters:** `p`, `steps`, `stepsize`, `lam`, `alpha`, `k`, `shots_train`, `shots_sample`, and `seed`; select `mixer="x"`.

**Guarantees and limits:** X mixing explores all cardinalities. A penalty discourages invalid selections; sampled outputs still require feasibility checks, and top-K projection is postprocessing. Finite-depth QAOA has no global optimality guarantee. The implementation also constructs full state probabilities, limiting simulator scale.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio import QAOAConfig, run_qaoa

cfg = QAOAConfig(mixer="x", lam=4.0, alpha="auto", k=2, p=1, steps=10, shots_sample=256, seed=0)
result = run_qaoa(mu, Sigma, cfg)
candidate = result.x_topk
```
