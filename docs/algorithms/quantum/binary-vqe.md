# Binary VQE

**Track:** binary asset selection. **Method ID:** `binary-vqe`.

Trains a parameterized circuit with Adam to minimize the expectation of the Ising Hamiltonian encoding the binary mean–variance objective plus alpha * (sum(x) - K)^2. Final measurements provide sampled selections, marginal probabilities, and a top-K projection.

**Dependencies and execution:** PennyLane and the package's core dependencies.
The default device is the `default.qubit` simulator. Analytic training is the
default; finite-shot training can be requested. Device selection alone does not
demonstrate hardware execution or quantum advantage.

**Key parameters:** `depth`, `ansatz`, `steps`, `stepsize`, `lam`, `alpha`, `k`, `shots_train`, `shots_sample`, `seed`, and optional restart, energy-normalization, best-iterate, and learning-rate-decay controls.

**Guarantees and limits:** The ansatz does not preserve cardinality. The penalty influences training; top-K projection enforces K by postprocessing. Sampled portfolios may be infeasible and `x_best_feasible` can be absent. Training is a variational heuristic with no global optimality guarantee. The runner supports 1 <= K <= n.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio import BinaryVQEConfig, run_binary_vqe

cfg = BinaryVQEConfig(lam=4.0, alpha="auto", k=2, steps=10, shots_sample=256, seed=0)
result = run_binary_vqe(mu, Sigma, cfg)
candidate = result.x_topk
```
