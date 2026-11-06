# Variational quantum allocator

**Track:** long-only fractional allocation. **Method ID:** `variational-allocator`.

Uses circuit Z expectations to parameterize a long-only allocation and evaluates the mean–variance cost classically. Adam optimizes the circuit parameters through this allocation map.

**Dependencies and execution:** PennyLane and the package's core dependencies.
The default device is the `default.qubit` simulator. Analytic training is the
default; finite-shot training can be requested. Device selection alone does not
demonstrate hardware execution or quantum advantage.

**Key parameters:** `depth`, `ansatz`, `steps`, `stepsize`, `lam`, `shots`, and `seed`. The default `ry` ansatz is a product of single-qubit rotations.

**Guarantees and limits:** This custom allocator is not Hamiltonian-expectation VQE. Its objective is convex in weights for positive semidefinite covariance, but circuit-parameter training has no global optimality guarantee. The default product circuit can be simulated classically without an exponential state vector; its use does not establish quantum advantage.

The allocation map uses $s_i=(1-\langle Z_i\rangle)/2$ and
$w_i=(s_i+\epsilon/n)/(\sum_j s_j+\epsilon)$, with $\epsilon=10^{-12}$.
Zero measured mass gives uniform weights. Small numerical expectation overshoots
are clipped at the physical boundaries; the normalization remains differentiable
in the interior and allocations sum to one within floating-point tolerance.

Historical names `run_fractional_vqe`, `FractionalVQEConfig`, and
`fractional-vqe` remain compatibility aliases. The canonical CLI command is
`allocator`; the legacy command `fractional` is also accepted.

## Example

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio import VariationalAllocatorConfig, run_variational_allocator

cfg = VariationalAllocatorConfig(lam=4.0, steps=10, seed=0)
result = run_variational_allocator(mu, Sigma, cfg)
weights = result.weights
```
