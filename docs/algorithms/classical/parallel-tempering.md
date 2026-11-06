# Parallel tempering / replica exchange

**Track:** exact-K binary selection. **Method ID:** `parallel-tempering`.

`parallel_tempering_binary_baseline` implements replica-exchange Metropolis
search for the original binary mean–variance objective, without a cardinality
penalty. The method follows
[Hukushima and Nemoto's exchange Monte Carlo procedure](https://arxiv.org/abs/cond-mat/9512035),
adapted to exact-K portfolio swaps and a fixed temperature ladder. NumPy is the
only dependency. Replicas execute serially; the algorithm does not require
threads, multiprocessing, or quantum execution.

## Local moves and exchanges

Each temperature slot holds a feasible portfolio and its cached energy,
gradient, selected indices, and unselected indices. Every iteration proposes one
uniform selected/unselected exchange in each slot. For objective change
$\Delta=f(x')-f(x)$ and temperature $T$, accept with probability
$\min(1,\exp(-\Delta/T))$. The symmetric proposal preserves exactly K assets.
Quadratic swap deltas use the maintained gradient; accepted portfolios are
rescored using the original objective before considering them as incumbents.

Every `exchange_interval` iterations, attempt exchanges between disjoint adjacent
temperature slots. Alternate edges `(0,1), (2,3), ...` and `(1,2), (3,4), ...`.
With two replicas, attempt their single edge every exchange round.

For neighboring temperatures $T_i<T_j$, accept an exchange with probability

$$
\min\left(1,\exp\left[(1/T_i-1/T_j)(f(x_i)-f(x_j))\right]\right).
$$

This is the ratio of swapped to current product-Boltzmann weights. Exchange the
complete states and caches while keeping temperatures fixed. All acceptance
tests use log ratios and exponential random draws, avoiding exponentiation of
large energies. Exchanges rearrange already evaluated states; they do not create
new objective evaluations. The retained best portfolio spans every temperature
and restart, including initial states and states later left behind.

## Configuration and diagnostics

Defaults are `seed=0`, `restarts=16`, `max_iterations=1000`, `n_replicas=8`, and
`exchange_interval=10`. An iteration means **one proposal per replica**, so a
completed nontrivial run makes `restarts * max_iterations * n_replicas` local
proposals. It is not equivalent to the same iteration count of single-chain
simulated annealing. Use end-to-end budgets for performance comparisons.

The default ladder is geometric from `0.01*S` to `S`, where `S` is the largest
absolute swap cost from the top-return selection. A flat starting neighborhood
uses `S=1`; nonzero scales have a numerical floor of `100 * finfo(float).tiny`.
Scale construction evaluates `K*(n-K)` swap costs and uses no reference optimum.
The ladder remains fixed across all restarts; it does not implement adaptive
temperature tuning. Poor ladder choices can prevent useful exchanges.

Pass `temperatures` to supply exactly `n_replicas` strictly increasing, positive,
finite values. At least two replicas are required. All count parameters must be
positive integers. Across all replicas and restarts, initialization uses top
returns, then diagonal costs, then seeded random exact-K subsets. `restarts`
counts fresh groups of replica states sharing the fixed ladder.

The result has status `iteration_limit`, or `trivial` for K=0/K=n. Neither a
finite search nor a high acceptance rate certifies optimality or equilibrium.
`lower_bound` and `mip_gap` remain absent. Nonfinite evaluated swap costs or
portfolio energies raise an error.

`ClassicalResult.diagnostics` records the ladder and controls, completed restarts
and iterations, local proposals and accepts by temperature, exchange attempts
and accepts by adjacent edge, acceptance rates, original-objective evaluations,
and swap evaluations used to choose the default scale. Rates are `None` before
an attempt. The message includes the same diagnostics as JSON for comparison
notes; validation results expose `solver_diagnostics` directly.

An optional `candidate_callback(candidate, details)` receives independent copies
only on strict incumbent improvement. Its details include the source, restart,
iteration, temperature slot, and a copied diagnostic snapshot. Within an
iteration the snapshot can contain proposals from a partially completed sweep.
Callbacks do not alter the solver's random stream or retained state.

The budget API and CLI return final `solver_diagnostics` when completion metadata
arrives within the deadline. On timeout or failure that field remains empty;
delivered incumbents retain their partial diagnostic snapshots. The budget
includes startup, initialization, scale selection, search, callbacks, and
independent candidate checks. Budget and validation CSVs encode structured
diagnostics as JSON. Archived study outputs are unchanged.

## Examples

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import parallel_tempering_binary_baseline

result = parallel_tempering_binary_baseline(
    mu, Sigma, lam=4.0, k=2, seed=0,
    restarts=2, n_replicas=4, temperatures=[0.01, 0.05, 0.2, 1.0],
)
print(result.candidate, result.objective, result.diagnostics)
```

```bash
vqe-portfolio budget --method parallel-tempering \
  --mu "0.08,0.12,0.10" \
  --sigma "0.04,0.01,0;0.01,0.09,0.01;0,0.01,0.06" \
  --lam 4 --k 2 --seed 0 --budget-seconds 10

python scripts/generate_comparison_results.py --methods parallel-tempering,anneal \
  --asset-counts 4,8 --k 2 --seeds 0,1,2 --local-restarts 2 \
  --output results/tempering_summary.csv --trials-output results/tempering_trials.csv
```

The small-instance validation and fixed-deadline runners accept the same method
ID. Comparison and validation use every configured solver seed. The budget
interface uses solver defaults; the ordinary Python API exposes ladder controls.
