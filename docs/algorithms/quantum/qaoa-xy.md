# QAOA with XY exchange mixing

**Track:** binary asset selection. **Method ID:** `qaoa-xy`.

Alternates diagonal cost phases with an ordered sequence of paired IsingXX and IsingYY gates on ring or complete-graph edges. It starts from a fixed exact-K basis state or a uniform exact-K Dicke state and optimizes the layer angles using Adam.

Optional [CVaR training](cvar.md) minimizes a lower tail of penalized costs;
finite-shot CVaR uses SPSA instead of Adam.

**Dependencies and execution:** PennyLane and the package's core dependencies.
The default device is the `default.qubit` simulator. Analytic training is the
default; finite-shot training can be requested. Device selection alone does not
demonstrate hardware execution or quantum advantage.

**Key parameters:** `p`, `steps`, `stepsize`, `lam`, `k`, `xy_initial_state` (`basis` or `dicke`), `xy_topology` (`ring` or `complete`), shots, and `seed`; select `mixer="xy"`.

**Guarantees and limits:** Paired exchange gates preserve Hamming weight in ideal execution. The ordered edge product is not generally the exact exponential of the summed XY Hamiltonian because overlapping edges need not commute. Noise and readout errors can violate cardinality; feasibility still needs checking. There is no global optimality guarantee, and full state probabilities limit simulator scale.

Budget presets `qaoa-xy-untrained` and `qaoa-xy-trained` reuse this algorithm
with different training settings; they are not separate implementations.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio import QAOAConfig, run_qaoa

cfg = QAOAConfig(mixer="xy", xy_initial_state="dicke", xy_topology="ring", lam=4.0, k=2, p=1, steps=10, shots_sample=256, seed=0)
result = run_qaoa(mu, Sigma, cfg)
candidate = result.x_best_feasible
```
