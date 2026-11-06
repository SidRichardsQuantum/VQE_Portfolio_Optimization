# Algorithms

Concise descriptions of the algorithms implemented in this package, including
reference solvers and simple comparison baselines.

- [Quantum methods](quantum/index.md): Binary VQE, QAOA with X or XY mixing,
  and the variational quantum allocator.
- [Classical methods](classical/index.md): exact references, optimization
  backends, local and population searches, and allocation baselines.

## Portfolio objectives

Both tracks minimize risk minus expected return:

$$
f(v) = \lambda v^\top\Sigma v - \mu^\top v.
$$

Binary selection uses $v=x$, $x_i\in\{0,1\}$, and $\sum_i x_i=K$.
Fractional allocation uses $v=w$, $w_i\geq0$, and $\sum_i w_i=1$.
Binary optimization scores raw selections; reporting an equal-weight portfolio
on the chosen assets uses $x/K$ when $K>0$. Scores across these two tracks are
not directly comparable. See the [theory](../THEORY.md) for the full formulation.

## Inputs for the examples

Run this setup before the Python example on an algorithm page:

```python
import numpy as np

mu = np.array([0.08, 0.12, 0.10])
Sigma = np.array([
    [0.04, 0.01, 0.00],
    [0.01, 0.09, 0.01],
    [0.00, 0.01, 0.06],
])
```

The examples use risk aversion `lam=4.0` and binary cardinality `k=2`.
Reference functions return `(candidate, objective)`; classical solver wrappers
return `ClassicalResult`, whose candidate can be absent on failure.
Quantum functions return their method-specific result objects.

Inspect solver status and independently check feasibility before comparing
objectives. Native solver time limits exclude some Python overhead; the common
`run_budgeted` interface measures startup-inclusive execution instead. See
the [usage guide](../USAGE.md) for CLI commands, budgets, and study runners.

Method IDs below identify study or budget selections where supported; they are
not individual top-level CLI subcommands. Historical names remain aliases rather
than separate algorithms.

```{toctree}
:maxdepth: 2

quantum/index
classical/index
```
