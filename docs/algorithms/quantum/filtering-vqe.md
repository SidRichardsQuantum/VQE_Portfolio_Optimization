# Filtering VQE (F-VQE)

**Track:** exact-K binary selection. **Method ID:** `f-vqe`.

`run_fvqe` implements the normalized parameter-shift procedure of Amaro et al.,
[Filtering variational quantum algorithms for combinatorial optimization](https://arxiv.org/abs/2106.10055),
Eqs. (6)–(7). Every iteration takes one gradient-descent step toward a freshly
filtered version of the current state. It does not run Adam on a transformed
energy expectation or reuse the CVaR objective.

For the current parameters $\theta$, the frozen target is
$F|\psi(\theta)\rangle/\sqrt{\langle F^2\rangle_\theta}$. The distance gradient
and update are

$$
g_j=-\frac{\langle F\rangle_{\theta+\pi e_j/2}
                 -\langle F\rangle_{\theta-\pi e_j/2}}
                {4\sqrt{\langle F^2\rangle_\theta}},
\qquad \theta\leftarrow\theta-\text{stepsize}\,g.
$$

Both implemented filters are positive and decrease with energy:

- `filter_type="inverse"`: $f(\widetilde E;\tau)=\widetilde E^{-\tau}$.
- `filter_type="exponential"`: $f(\widetilde E;\tau)=\exp(-\tau\widetilde E)$.

The original penalized cost is
$E(x)=\lambda x^\top\Sigma x-\mu^\top x+\alpha(\sum_i x_i-K)^2$.
Coefficient-wise QUBO bounds $L,U$ give the affine scale
$\widetilde E=\epsilon+(1-\epsilon)(E-L)/(U-L)$, with
`spectrum_floor` $\epsilon=0.001$ by default. This makes inverse filtering valid
even for negative costs, without finding an optimum or enumerating the spectrum
to obtain bounds. A constant Hamiltonian produces an identity filter. Moments
are evaluated in log space. The energy trace retains original penalized units.

## Filter strength

By default `adapt_tau=True` selects strength using the gradient norm. Starting
at `tau`, it doubles trial strengths up to `tau_max`; a threshold crossing is
refined by `tau_search_steps` bisections. It retains the tried strength with the
closest norm at or below `gradient_target`. If the threshold is unreachable,
selection remains bounded by `tau_max`. This is a bounded version of the paper's
heuristic; nonmonotone gradient norms need not yield a global best strength.
All trial strengths reuse the same measured distributions. Fixed-strength
execution uses `adapt_tau=False` and `tau` directly.

Defaults are `tau=1`, `tau_max=64`, `gradient_target=0.1`, and
`tau_search_steps=20`. The selected strength and actual gradient norm are
recorded at every update. Filter choice and strength affect the update direction;
the code does not simply rescale a gradient to the target norm.

## Execution and returned candidates

Core dependencies suffice. The default device is `default.qubit`.

- `shots_train=None`: evaluate computational-basis probabilities. This path
  enumerates costs and retains shifted distributions, requiring exponential
  storage in the number of assets. It is intended for small simulator studies.
- Positive `shots_train`: execute computational-basis sampling circuits and
  evaluate filter moments only on measured bitstrings. No spectrum enumeration,
  statevector access, Hadamard test, or dense filter matrix is required by the
  algorithm. A simulator backend still has its own state-storage costs.
- All three binary ansätze are supported. Each update uses $2m+1$ training
  circuit evaluations for $m$ rotation parameters, with one additional final
  training evaluation: `steps * (2*m + 1) + 1`. Adaptive strength selection adds
  classical processing, but no circuit evaluations.

`energy_trace` includes the initial state (step 0), logging checkpoints, and the
final post-update state. It records mean penalized energy, rather than the
changing state-distance objective. Finite-shot energy estimates use the same
unshifted samples as the filter normalization.

Final candidate sampling uses `shots_sample`. The best feasible sampled
portfolio is exposed as `x_best_feasible`; callers use `x_topk` as an explicit
fallback when no feasible sample exists. Analytic inclusion probabilities use
the cached final training distribution; finite-shot inclusion probabilities use
the final candidate batch. Candidate callbacks publish at step 0, logging
checkpoints, and completion, with separate sample batches and a source label.
Their cost is included in budgeted execution. Training and candidate sampling
use separate seeded device streams, so callbacks do not change training.
Devices must support the selected measurements and a `seed` constructor option.

Neither exact filtering nor a variational projection guarantees that a finite
run returns an optimum. Finite-shot estimates and large learning rates can
increase energy. Compare feasible candidates using the shared independent
evaluator and matched end-to-end budgets.

## Examples

Use the [shared inputs](../index.md#inputs-for-the-examples).

```python
from vqe_portfolio import FVQEConfig, PortfolioProblem, run_fvqe

cfg = FVQEConfig(k=2, lam=4, steps=25, filter_type="inverse")
result = run_fvqe(mu, Sigma, cfg)
candidate = result.x_best_feasible
if candidate is None:
    candidate = result.x_topk
problem = PortfolioProblem(mu, Sigma, cfg.lam, "binary_qubo", cfg.k)
print(problem.evaluate(candidate))
print(result.tau_trace, result.gradient_norm_trace)
```

```bash
vqe-portfolio f-vqe --mu "0.1,0.2,0.15" \
  --sigma "0.05,0,0;0,0.06,0;0,0,0.04" --k 2 --lam 4 \
  --filter exponential --fixed-tau --tau 1 --shots-train 1024 --steps 25
```

`f-vqe-data` accepts the same solver options with the market-data arguments.
Both commands support `--budget-seconds`. The generic budget API also accepts
`run_budgeted(problem, "f-vqe", 10, config=cfg)`.

For synthetic CSV comparisons use
`python scripts/generate_comparison_results.py --methods classical,f-vqe`.
The small-instance validation and fixed-deadline benchmark runners accept the
`f-vqe` method ID. These runners use adaptive inverse filtering; use the Python
API or solver CLI to compare other filtering settings. Historical saved studies
are unchanged and do not contain F-VQE results.
