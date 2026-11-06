# CVaR training for Binary VQE and QAOA

**Track:** binary asset selection. **Setting:** `training_objective="cvar"`.

CVaR changes how circuit outcomes are aggregated during training. Instead of
minimizing the mean penalized cost, minimize the average of the lowest-cost
`cvar_alpha` fraction of probability mass. The circuits, portfolio objective,
cardinality checks, and final candidate evaluation remain the same.

For bitstrings sorted by penalized cost $E_1\leq\cdots\leq E_m$, with
probabilities $p_i$, the lower-tail objective is

$$
C_a = \frac{1}{a}\sum_i E_i\min\left(p_i,
              \max\left(a-\sum_{j<i}p_j,0\right)\right),\qquad 0<a\leq1.
$$

The observation crossing the quantile contributes only the mass needed to reach
`a`. Finite-shot training applies the same rule to equally weighted observations;
it does not round the tail size up to a whole number of shots. Infeasible
bitstrings retain their squared-cardinality penalty during training.

This is the lower tail of the **optimization-cost distribution over measured
bitstrings**, not the upper tail of financial losses across market scenarios.
It does not change mean–variance selection into financial CVaR optimization.

## Execution and optimization

- `shots_train=None`: compute all basis-state probabilities, sort constant costs
  once, and differentiate the probability-weighted tail with Adam. CVaR is
  piecewise differentiable; quantile boundaries are nonsmooth. This path uses
  `O(n * 2**n)` setup storage and is intended for small simulator problems.
- Positive `shots_train` and `cvar_alpha < 1`: sample bitstrings, score them
  classically, and use SPSA. No gradient is taken through sampling or sorting.
  Each update uses two perturbed cost evaluations; logging and best-iterate
  selection require additional evaluations. The perturbation magnitude is
  `0.1/(t+1)**0.101`; the learning rate is
  `stepsize/(t+1)**0.602/(1+stepsize_decay*t)` at zero-based update `t`.
  QAOA has no additional `stepsize_decay` setting and uses zero decay.
- `cvar_alpha=1`: use the existing mean-cost QNode and Adam, preserving the
  expectation-training trajectory for the same configuration and seed.

The default remains `training_objective="expectation"`. The tail fraction
defaults to `0.25` and must be finite and in `(0, 1]`. Small tails can flatten
the analytic objective once enough mass reaches the best costs, and increase
noise in sampled training. There is no guarantee of improved portfolio quality.
Use 10%, 25%, and 100% as experiment settings, with matched execution budgets.

CVaR applies to both QAOA mixers and both solvers' lambda sweeps. Binary VQE
normalization, best-iterate selection, and restart selection use the configured
cost. `energy_trace`/`cost_trace` and `restart_energies` record CVaR costs in
original penalized objective units when enabled. The historical field names are
retained. Results identify `training_objective`, `cvar_alpha` (null for
expectation), and `training_optimizer` (`adam` or `spsa`).

Budgeted execution accepts the same configuration and includes CVaR setup,
optimization and sampling costs. Finite-shot CVaR training avoids enumerating
all bitstrings; QAOA's existing final marginal/probability reporting still
constructs the full probability vector.

## Examples

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio import BinaryVQEConfig, QAOAConfig, run_binary_vqe, run_qaoa

config = BinaryVQEConfig(
    k=2, lam=4.0, alpha="auto", steps=25,
    training_objective="cvar", cvar_alpha=0.25,
)
result = run_binary_vqe(mu, Sigma, config)

config = QAOAConfig(
    k=2, lam=4.0, mixer="xy", xy_initial_state="dicke", p=1,
    steps=25, stepsize=0.05, shots_train=1024,
    training_objective="cvar", cvar_alpha=0.25,
)
result = run_qaoa(mu, Sigma, config)
```

```bash
vqe-portfolio binary --mu "0.1,0.2,0.15" \
  --sigma "0.1,0,0;0,0.2,0;0,0,0.1" --k 2 --lam 4 \
  --training-objective cvar --cvar-alpha 0.25

vqe-portfolio qaoa --mu "0.1,0.2,0.15" \
  --sigma "0.1,0,0;0,0.2,0;0,0,0.1" --k 2 --lam 4 \
  --mixer xy --xy-initial-state dicke --stepsize 0.05 \
  --training-objective cvar --cvar-alpha 0.25 --shots-train 1024
```

The same flags are available on `binary-data` and `qaoa-data`; add
`--budget-seconds` to the solver commands for budgeted execution. CVaR is a
training setting, not a separate method ID, and is not applied to the
variational allocator.

Reference: Barkoutsos et al.,
[Improving Variational Quantum Optimization using CVaR](https://arxiv.org/abs/1907.04769).
