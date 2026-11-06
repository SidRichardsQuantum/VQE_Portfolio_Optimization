# Frank–Wolfe with away steps

**Track:** long-only fractional allocation. **Method ID:** `frank-wolfe`.

`frank_wolfe_baseline` implements Algorithm 1 of
[Lacoste-Julien and Jaggi (2015)](https://papers.neurips.cc/paper_files/paper/2015/hash/c058f544c737782deacefa532d9add4c-Abstract.html)
on the probability simplex. It minimizes the existing fractional objective
$f(w)=\lambda w^\top\Sigma w-\mu^\top w$ using NumPy only.

## Algorithm

Start at the simplex vertex for the largest expected return, breaking ties by
lowest index. At each iteration, compute $g=2\lambda\Sigma w-\mu$ and:

1. Choose the forward vertex $e_s$ with the smallest gradient component and the
   away vertex $e_v$ with the largest gradient component among positive weights.
2. Compare the forward gap $g^\top w-g_s$ and away gap $g_v-g^\top w$.
   Choose the forward direction $d=e_s-w$ on ties; otherwise choose the direction
   with the larger gap. Away steps require at least two active vertices.
3. Bound the step by $\gamma_{\max}=1$ for forward steps or
   $\gamma_{\max}=w_v/(1-w_v)$ for away steps. For positive curvature, use the
   exact quadratic line minimizer
   $\gamma=\min(\gamma_{\max},-g^\top d/(2\lambda d^\top\Sigma d))$.
   For nonpositive numerical curvature, use the endpoint.
4. Update the convex coefficients. A maximal away step sets the removed vertex's
   weight exactly to zero. Renormalization corrects floating-point sum drift;
   there is no simplex projection.

The active set is represented directly by positive simplex weights. Sparse
iterates do not enforce a prescribed number of assets. This solver does not
solve the separate exact-K binary problem.

## Controls, stopping, and execution

Defaults are `max_iterations=10000` and `tolerance=1e-8`. The iteration limit
must be a positive integer and the tolerance finite and positive. Covariance
eigenvalues below `-1e-10` are rejected. Singular PSD inputs and linear objectives
are supported without matrix inversion or division by zero curvature.

`converged` means the returned portfolio's numerical forward gap is at most the
tolerance. `iteration_limit` retains the best feasible incumbent after the update
limit. `numerical_limit` retains it when floating-point stagnation or invalid
numerical updates prevent progress, or when a later iterate meets the gap
criterion but the retained best-scoring portfolio does not. Exact line search
in arithmetic does not eliminate numerical stopping limits.

The result message records completed forward, away, and drop steps, the returned
active-set size, and `first_order_gap`. The numerical gap is not independently
certified; `lower_bound` and `mip_gap` remain absent.

An optional `candidate_callback(candidate, details)` receives independent copies
of strictly improving incumbents, including the initial vertex. Details identify
the source and iteration; update callbacks also include `step_type`, `atom`,
`step_size`, and `max_step_size`. The solver is deterministic and uses no seeds.
Comparison and validation runners execute it once per problem instance.

The common budget interface includes input validation, the PSD check, gradient
calculations, line searches, callbacks, and process startup. A deadline can retain
an incumbent received and independently validated before expiry. The solver may
finish before its allowance when it converges or reaches its iteration limit.

## Examples

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio.baselines import frank_wolfe_baseline

result = frank_wolfe_baseline(mu, Sigma, lam=4.0, tolerance=1e-8)
print(result.candidate, result.status, result.message)
```

The budget CLI requires no cardinality argument:

```bash
vqe-portfolio budget --method frank-wolfe \
  --mu "0.08,0.12,0.10" \
  --sigma "0.04,0.01,0;0.01,0.09,0.01;0,0.01,0.06" \
  --lam 4 --budget-seconds 10
```

Compare against PGD and optional OSQP on the same fractional objective:

```bash
python scripts/generate_comparison_results.py --methods frank-wolfe,pgd,osqp \
  --asset-counts 4,8 --output results/fractional_comparison.csv \
  --trials-output results/fractional_trials.csv
```

The small-instance validation and fixed-deadline runners also accept
`frank-wolfe`. OSQP needs the optional classical dependencies; Frank–Wolfe and
PGD do not. Archived study results do not contain the new method.
