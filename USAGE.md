# Usage

This document shows how to use the `vqe_portfolio` package **without notebooks**, directly from Python.
Notebooks in this repository are thin clients around the same API.

---

## Command-Line Interface (CLI)

The package provides a first-class CLI for running quantum portfolio optimization **without writing Python code**.

After installation, the CLI is available as:

```bash
vqe-portfolio --help
```

You can also invoke it via:

```bash
python -m vqe_portfolio --help
```

---

### Binary VQE (Asset Selection)

Run a binary VQE to select exactly **K assets** under a cardinality constraint.

#### Inline synthetic data

```bash
vqe-portfolio binary \
  --mu "0.10,0.12,0.07,0.09" \
  --sigma "0.02,0.00,0.00,0.00;0.00,0.02,0.00,0.00;0.00,0.00,0.02,0.00;0.00,0.00,0.00,0.02" \
  --k 2 \
  --lam 4.0 \
  --alpha 2.0 \
  --ansatz ry_cz \
  --steps 80 \
  --out binary_result.json
```

This produces a JSON file containing:

- optimized circuit parameters
- inclusion probabilities
- Top-K selection
- sampled bitstrings and counts
- optimization trace

Supported Binary VQE ansatz values are `ry_cz`, `ry_rz_cz`, and `strongly_entangling`.

---

### Filtering VQE (F-VQE)

F-VQE solves the binary selection problem using filtered-state distance
gradients. The [algorithm guide](docs/algorithms/quantum/filtering-vqe.md)
describes its update rule, filters, normalization, and measurement costs.

```python
from vqe_portfolio import FVQEConfig, run_fvqe

result = run_fvqe(mu, Sigma, FVQEConfig(k=2, lam=4, steps=25))
```

```bash
vqe-portfolio f-vqe \
  --mu "0.10,0.12,0.07,0.09" \
  --sigma "0.02,0,0,0;0,0.02,0,0;0,0,0.02,0;0,0,0,0.02" \
  --k 2 --lam 4 --alpha auto --filter inverse --steps 25 \
  --out fvqe_result.json
```

Adaptive filter strength is the default. Use `--fixed-tau --tau 1` for a fixed
strength, `--filter exponential` for exponential filtering, and
`--shots-train 1024` for empirical filter moments. Adaptive search options are
`--gradient-target`, `--tau-max`, and `--tau-search-steps`.
`--spectrum-floor` controls the positive affine energy range (default `0.001`).
F-VQE has its own config and update loop; it does not accept CVaR or Adam options.

Use `f-vqe-data --tickers "AAPL,MSFT,NVDA" --k 2 --lam 4` for market data,
with the same date and data options as the other solvers. Both commands support
`--budget-seconds`. Budget and study method IDs use `f-vqe`.

---

### QAOA (Binary Asset Selection)

Solve the same constrained mean–variance problem using the **Quantum Approximate Optimization Algorithm (QAOA)**.

#### Inline synthetic data

```bash
vqe-portfolio qaoa \
  --mu "0.10,0.12,0.07,0.09" \
  --sigma "0.02,0.00,0.00,0.00;0.00,0.02,0.00,0.00;0.00,0.00,0.02,0.00;0.00,0.00,0.00,0.02" \
  --k 2 \
  --lam 4.0 \
  --alpha 2.0 \
  --p 2 \
  --steps 80 \
  --mixer x \
  --out qaoa_result.json
```

Key options:

| option           | meaning                                       |
| ---------------- | --------------------------------------------- |
| `--p`            | QAOA circuit depth                            |
| `--mixer`        | `"x"` (standard) or `"xy"` (constraint-aware) |
| `--shots-sample` | number of measurement samples                 |
| `--seed`         | reproducibility                               |

Output JSON includes:

- optimized γ and β parameters
- marginal inclusion probabilities
- Top-K selection
- sampled bitstrings and counts
- best feasible candidate solution
- optimization trace

---

### Variational quantum allocator (Continuous Allocation)

The canonical Python API is `run_variational_allocator`,
`VariationalAllocatorConfig`, `VariationalAllocatorResult`, and
`allocator_lambda_sweep`. The historical names `run_fractional_vqe`,
`FractionalVQEConfig`, `FractionalVQEResult`, and `fractional_lambda_sweep` are
exact compatibility aliases. CLI commands `allocator` and `allocator-data`
accept `fractional` and `fractional-data` as historical aliases. New comparison
rows use "Variational quantum allocator"; `variational-allocator` is the method
ID, and `fractional-vqe` remains accepted by comparison, validation, and budget
runners. Defaults include only the canonical ID so aliases do not duplicate runs.

This is a circuit-based allocation model with a classical portfolio objective,
not conventional Hamiltonian VQE. For masses `s=(1-<Z>)/2`, weights are
`(s+epsilon/n)/(sum(s)+epsilon)`, with `epsilon=1e-12`. Zero measured mass yields
uniform weights. The map is differentiable on physical interiors; numerical
expectation overshoots within `1e-8` are clipped at the measurement boundaries.
Invalid/nonfinite expectations are rejected. Finite-shot runs remain stochastic.
The denominator-only historical normalisation is no longer used, so save new
experiments separately and retain historical artifacts with their source hashes.

Solve the long-only mean–variance problem on the simplex.

#### Using an input JSON file

Create `input.json`:

```json
{
  "mu": [0.10, 0.12, 0.07],
  "sigma": [
    [0.04, 0.01, 0.00],
    [0.01, 0.09, 0.02],
    [0.00, 0.02, 0.03]
  ]
}
```

Run:

```bash
vqe-portfolio allocator \
  --input input.json \
  --lam 5.0 \
  --ansatz ry \
  --depth 1 \
  --steps 100 \
  --out fractional_result.json
```

The output JSON includes:

- optimized circuit parameters
- portfolio weights
- cost trace

Supported Variational quantum allocator ansatz values are `ry`, `ry_cz`, and `ry_rz_cz`.

---

### Input formats

You may provide inputs in either form:

#### Inline

```bash
--mu "0.1,0.2,0.3"
--sigma "1,0.1;0.1,2"
```

Note: when providing `--sigma` inline, the argument must be quoted because `;` is interpreted by the shell as a command separator.

Example:

```bash
--sigma "0.1,0.0;0.0,0.2"
```

#### JSON

```json
{
  "mu": [...],
  "sigma": [...]
}
```

JSON is recommended for reproducibility and larger problem sizes.

---

### Reproducibility via CLI

All CLI commands respect the same reproducibility controls as the Python API:

```bash
--seed 0
```

For repeatable benchmark artifacts, use the committed results-generation script:

```bash
python scripts/generate_comparison_results.py
```

It writes a summary CSV and a per-seed trial CSV under `results/`. The defaults are intentionally compact; expand them with `--asset-counts`, `--seeds`, `--steps`, and `--methods`.

`notebooks/Benchmark_Comparison.ipynb`, `notebooks/Real_Data_Comparison.ipynb`, and `notebooks/Ansatz_Comparison.ipynb` are pure-package clients for the same comparison workflows. They display CSV tables and save comparison plots under `notebooks/images/`.

---

### Relationship to the Python API

The CLI is a **thin client** over the same public API used by notebooks and scripts:

- `vqe-portfolio binary` → `run_binary_vqe`
- `vqe-portfolio qaoa` → `run_qaoa`
- `vqe-portfolio allocator` → `run_variational_allocator`
- benchmark notebooks/scripts → `vqe_portfolio.research.comparison` workflows and `vqe_portfolio.benchmark` evaluation helpers and public ansatz config options

No logic is duplicated.

---

## 1. Minimal Example (Synthetic Data)

```python
import numpy as np
from vqe_portfolio import run_variational_allocator, VariationalAllocatorConfig

mu = np.array([0.10, 0.12, 0.07])

Sigma = np.array([
    [0.04, 0.01, 0.00],
    [0.01, 0.09, 0.02],
    [0.00, 0.02, 0.03],
])

cfg = VariationalAllocatorConfig(
    lam=5.0,
    ansatz="ry",
    depth=1,
    steps=100,
    stepsize=0.25,
    shots=None,
)

res = run_variational_allocator(mu, Sigma, cfg)

print(res.weights)
```

This solves the long-only mean–variance problem on the simplex **by construction**.

---

## 2. Binary VQE (Asset Selection)

```python
import numpy as np
from vqe_portfolio import run_binary_vqe, BinaryVQEConfig

mu = np.array([0.10, 0.12, 0.07, 0.09])
Sigma = 0.02 - np.eye(4)

cfg = BinaryVQEConfig(
    k=2,
    lam=4.0,
    alpha=2.0,
    ansatz="ry_cz",
    depth=2,
    steps=80,
)

res = run_binary_vqe(mu, Sigma, cfg)

print("Top-K selection:", res.x_topk)
print("Probabilities:", res.x_prob)
```

Binary VQE solves the QUBO → Ising → VQE formulation of the cardinality-constrained problem.
Use `ansatz="ry_cz"`, `ansatz="ry_rz_cz"`, or `ansatz="strongly_entangling"` to compare circuit families.

---

## 3. QAOA (Binary Asset Selection)

```python
import numpy as np
from vqe_portfolio import run_qaoa, QAOAConfig

mu = np.array([0.10, 0.12, 0.07, 0.09])
Sigma = 0.02 - np.eye(4)

cfg = QAOAConfig(
    k=2,
    lam=4.0,
    alpha=2.0,
    p=2,
    steps=80,
    mixer="x",
)

res = run_qaoa(mu, Sigma, cfg)

print("Top-K selection:", res.x_topk)
print("Mode bitstring:", res.x_mode)
print("Probabilities:", res.x_prob)
```

QAOA solves the same QUBO / Ising formulation as Binary VQE but uses an alternating operator ansatz:

$$
U(\boldsymbol{\gamma},\boldsymbol{\beta})
=
\prod_{\ell=1}^p
e^{-i\beta_\ell H_M}
e^{-i\gamma_\ell H_C}
$$

where:

- $H_C$ is the portfolio cost Hamiltonian
- $H_M$ is the mixer Hamiltonian

Two mixers are available:

| mixer  | behaviour                                   |
| ------ | ------------------------------------------- |
| `"x"`  | standard transverse-field mixer             |
| `"xy"` | better respects fixed-cardinality structure |

Result fields include:

- `x_prob` — marginal inclusion probabilities
- `x_topk` — deterministic Top-K projection
- `x_mode` — most frequently sampled bitstring
- `x_best_feasible` — best sampled feasible portfolio
- `state_probs` — full computational basis distribution

---

## 4. λ-Sweeps and Efficient Frontiers

### Fractional frontier

```python
from vqe_portfolio import allocator_lambda_sweep
from vqe_portfolio.frontier import fractional_frontier_from_allocs
from vqe_portfolio.types import LambdaSweepConfig

cfg = VariationalAllocatorConfig(lam=1.0, steps=80)

sweep = LambdaSweepConfig(
    lambdas=[0.5, 1.0, 2.0, 5.0, 10.0],
    steps_per_lambda=60,
    warm_start=True,
)

res = allocator_lambda_sweep(mu, Sigma, cfg, sweep)

frontier = fractional_frontier_from_allocs(
    mu,
    Sigma,
    res.lambdas,
    res.allocs_by_lambda,
)
```

The resulting `Frontier` object contains risk, return, λ, and portfolio weights.

---

### QAOA λ-sweep

```python
from vqe_portfolio import qaoa_lambda_sweep
from vqe_portfolio.types import QAOAConfig, LambdaSweepConfig

cfg = QAOAConfig(
    k=2,
    lam=1.0,
    p=2,
    steps=60,
)

sweep = LambdaSweepConfig(
    lambdas=[0.5, 1.0, 2.0, 5.0],
    steps_per_lambda=40,
    warm_start=True,
)

res = qaoa_lambda_sweep(mu, Sigma, cfg, sweep)

print(res.lambdas)
print(res.probs_by_lambda)
```

Produces marginal inclusion probabilities across risk-aversion values.

---

## 5. Real Market Data

Requires:

```bash
pip install "vqe-portfolio[data]"
```

```python
from vqe_portfolio import get_stock_data, run_variational_allocator
from vqe_portfolio.types import VariationalAllocatorConfig

mu, Sigma, prices = get_stock_data(
    ["AAPL", "MSFT", "GOOGL", "AMZN"],
    start="2024-01-01",
    end="2025-01-01",
    shrink="lw",
)

cfg = VariationalAllocatorConfig(
    lam=5.0,
    steps=100,
)

res = run_variational_allocator(
    mu.values,
    Sigma.values,
    cfg,
)

print(res.weights)
```

---

## 6. Reproducibility

```python
from vqe_portfolio.utils import set_global_seed

set_global_seed(0)
```

All optimization loops and random initializations respect the global seed.

For method-level repeatability reports, regenerate:

```bash
python scripts/generate_comparison_results.py
```

This records aggregate objective statistics in `results/generated_comparison_summary.csv` and per-seed trial rows in `results/generated_repeatability_trials.csv`.

Comparison CSVs distinguish `objective_family` from `reported_weighting`: binary candidates are scored on the original mean-variance objective using raw bitstrings, while binary risk/return uses equal-weight selected portfolios. Penalty energy is a separate diagnostic. Binary and fractional objective values must not be ranked together.

---

## 7. Notebooks as Clients

All notebooks in `notebooks/` simply:

- import the public API
- call `run_binary_vqe`, `run_qaoa`, or `run_variational_allocator`
- generate plots via `vqe_portfolio.plotting`
- regenerate or display comparison CSVs from the package clients

They contain **no core logic**.

---

## 8. What This Package Is (and Is Not)

**This package is:**

- A research-grade quantum optimization toolkit
- Deterministic, testable, and CI-validated
- Designed for experimentation and extension

**This package is not:**

- A production trading system
- A performance-optimized classical solver
- A claim of quantum advantage

---

For theory, see `THEORY.md`.

---

## Author

**Sid Richards**

LinkedIn:
[https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

GitHub:
[https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

---

## License

MIT License — see [LICENSE](https://github.com/SidRichardsQuantum/VQE_Portfolio_Optimization/blob/main/LICENSE)

## Strong Classical Baselines

Install the optional solver dependencies:

```bash
pip install -e ".[classical]"
```

Run the stronger baselines without exponential enumeration or quantum simulation:

```bash
python scripts/generate_comparison_results.py \
  --methods classical-solvers --asset-counts 8,16,32 --k 4 \
  --seeds 0,1,2 --solver-time-limit 60 --local-restarts 16 \
  --output results/strong_classical_summary.csv \
  --trials-output results/strong_classical_trials.csv
```

The methods solve the same objectives as the existing comparison tracks:

| Method | Formulation | What the result means |
| --- | --- | --- |
| Classical convex Markowitz (OSQP) | Convex quadratic objective with nonnegative weights summing to one, via CVXPY | Numerical optimum; tiny negative weights are clipped and weights renormalised before evaluation |
| Classical mixed integer (HiGHS) | Exact-K binary selection with a sparse linearisation of pairwise products, via SciPy | Incumbent, solver status, lower bound, and relative MIP gap; a limit status does not certify optimality |
| Classical quadratic integer (SCIP) | Exact-K binary selection with a quadratic objective epigraph, via PySCIPOpt | Incumbent, native status, numerical lower bound, and SCIP relative gap; streams feasible incumbents during solving |
| Classical QP relaxation + rounding | Box relaxation with sum(z)=K via CVXPY/OSQP, top-K rounding, then swap refinement | Deterministic binary heuristic; numerical relaxation diagnostics are not certified bounds |
| Classical multistart swap search | Exact-K selection, best-improvement swaps from top-return, diagonal-cost, and random starts | Seeded heuristic with no global optimality guarantee |
| Classical simulated annealing | Exact-K selection with random swaps and Metropolis acceptance under geometric cooling | Seeded heuristic that can accept uphill moves; retains the best feasible portfolio |
| Classical tabu search | Exact-K selection with best-admissible swaps, reverse-move memory, and aspiration for a new best objective | Seeded heuristic that can move uphill; retains the best feasible portfolio |
| Classical memetic genetic algorithm | Exact-K population search with feasible crossover, swap mutation, elitism, immigration, and local refinement | Seeded heuristic; retains the best feasible portfolio across populations |
| Classical iterated local search | Exact-K perturbations followed by swap refinement and improvement-only acceptance | Seeded heuristic; retains the best feasible portfolio across restarts |
| Classical GRASP | Covariance-aware randomised greedy construction followed by swap refinement | Seeded heuristic; publishes only complete exact-K portfolios |
| Classical variable neighborhood search | Exact-K shaking across increasing exchange sizes, followed by swap refinement | Seeded heuristic; resets to the smallest neighbourhood after improvement |
| Classical projected gradient descent | Convex long-only simplex objective with Euclidean projection and backtracking | NumPy iterative fractional comparator; numerical first-order stopping diagnostics, no certified bound |

The HiGHS formulation introduces a continuous variable `y_ij` per pair, constrained by `y_ij <= x_i`, `y_ij <= x_j`, and `y_ij >= x_i + x_j - 1`, with all variables bounded in `[0, 1]` and `x` binary. This represents both positive and negative covariance couplings exactly. Its model size grows quadratically; it is a stronger general-purpose comparator, not a claim to be the best possible classical solver.

`--solver-time-limit` limits each HiGHS/SCIP solve only. Reported runtime includes the entire solver helper call, including model construction and candidate cleanup; it can exceed that limit. OSQP uses accuracy tolerances, and the NumPy heuristics use restart/generation/iteration limits, not that time limit. For end-to-end deadlines use `run_budgeted` or the budget runner shown below.

`--seeds` controls independent stochastic classical runs as well as quantum runs. Each search run uses `--local-restarts` starts (minimum two when including swap search); for memetic search this counts independent populations, and for GRASP it counts complete constructions. `qp-round`, `pgd`, and `frank-wolfe` run once per instance and do not use seeds or restarts. `--k` fixes binary cardinality across requested sizes; PGD and Frank–Wolfe use simplex weights. Defaults remain compatible with the existing examples: `all` selects the original methods and excludes the stronger solver group. `classical-solvers` selects OSQP, HiGHS, SCIP, QP rounding, swap search, annealing, tabu search, memetic search, ILS, GRASP, VNS, PGD, and Frank–Wolfe. Use `--methods all,classical-solvers` to include them alongside the original small-instance references and quantum methods. Individual solver IDs `osqp`, `pgd`, `frank-wolfe`, `highs`, `scip`, `qp-round`, `swap`, `anneal`, `tabu`, `memetic`, `ils`, `grasp`, and `vns` are also accepted. Avoid `classical`/`all` for large instances because the old exact references enumerate subsets.

Run just annealing and tabu search with NumPy, without optional classical solver dependencies:

```bash
python scripts/generate_comparison_results.py \
  --methods anneal,tabu --asset-counts 8,16,32 --k 4 --seeds 0,1,2 \
  --output results/heuristics_summary.csv \
  --trials-output results/heuristics_trials.csv
```

The CSVs add `solver_status`, `lower_bound`, and `mip_gap`; legacy methods leave them blank. Trial rows retain unsuccessful solver returns with blank objective/portfolio values and zero feasibility. Summary quality statistics include feasible returns only, while runtime, trial count, and feasibility include failures. Inspect trial rows before drawing comparisons. A reported bound is the numerical solver bound, not an independent exact-arithmetic certificate. Missing dependencies or solver exceptions stop the run visibly rather than silently substituting a weaker solver.

Programmatic entry points are `convex_markowitz_baseline`, `mixed_integer_binary_baseline`, and `local_search_binary_baseline` in `vqe_portfolio.baselines`. They return a `ClassicalResult` with the candidate, recomputed original objective, termination status, and available bound metadata. The local-search helper also accepts `max_swaps` (default 1,000 per restart).

`relaxation_rounding_binary_baseline` solves `min lambda*z.T@Sigma@z - mu.T@z` with `0 <= z <= 1` and `sum(z)=K`, using CVXPY/OSQP. It retains the quadratic diagonal terms and raw selection scaling, and requires a PSD covariance. It selects the largest K relaxed entries (asset-index order for exact ties), then refines with improving swaps. Its returned binary portfolio has exactly K assets and is scored on the original binary objective.

```python
from vqe_portfolio.baselines import relaxation_rounding_binary_baseline

result = relaxation_rounding_binary_baseline(
    mu, Sigma, lam=4.0, k=4, max_swaps=1000, qp_max_iter=100000,
)
print(result.status, result.candidate, result.objective, result.message)
```

Install `.[classical]` or `.[markowitz]` for the existing CVXPY/OSQP dependencies. Native tolerances are `eps_abs=eps_rel=1e-9`, with warm starts disabled. Both iteration limits must be positive integers. `--solver-time-limit` does not limit this QP; use the budget interface for an end-to-end deadline. K=0 and K=n return the unique selection with status `trivial` without a native solve. Fractional QP output must be finite and satisfy the relaxed constraints within `1e-8`; tiny box-bound violations are clipped before rounding. Missing output retains the native QP status with no candidate, invalid output reports `invalid_relaxation`, and solver exceptions propagate without fallback.

Success reports `rounded_and_refined`, which certifies neither integer optimality nor convergence beyond the swap cap. Numerical relaxation status and recomputed fractional objective appear in the helper's `message`, comparison CSV notes, and validation's `solver_message`. `lower_bound` and `mip_gap` stay missing: a numerical primal relaxation value alone is not a certified lower bound. The optional `candidate_callback(candidate, details)` receives independent copies of improving binary incumbents after rounding and during refinement. No fractional candidate is published, so a budgeted run can have no candidate if its deadline expires before the QP returns.

```bash
python scripts/generate_comparison_results.py --methods qp-round \
  --asset-counts 8,16,32 --k 4 \
  --output results/qp_rounding_summary.csv --trials-output results/qp_rounding_trials.csv
```

`scip_binary_baseline` solves the same binary objective using the quadratic constraint `f(x) <= z`, minimising a continuous `z` with no lower bound and enforcing `sum(x) == k`. It uses binary diagonal identities and introduces no pairwise product variables in the Python formulation; SCIP may reformulate the problem internally. The unrestricted `z` allows negative objective values. Install `.[classical]` for the full solver group, or `.[scip]` for SCIP alone:

```bash
pip install -e ".[scip]"
python scripts/generate_comparison_results.py --methods scip \
  --asset-counts 8,16,32 --k 4 --solver-time-limit 60 \
  --output results/scip_summary.csv --trials-output results/scip_trials.csv
```

```python
from vqe_portfolio.baselines import scip_binary_baseline

result = scip_binary_baseline(mu, Sigma, lam=4.0, k=4,
                              time_limit=60, mip_rel_gap=0.0)
print(result.status, result.objective, result.lower_bound, result.mip_gap)
```

SCIP uses one solving thread and runs once per instance in validation and comparison tables. Its native statuses include `optimal`, `timelimit`, and `gaplimit`; a limit may return a feasible incumbent or no candidate. Returned selections are checked for numerical integrality and exact cardinality before the original objective is recomputed. Unavailable or infinite native bounds/gaps are reported as missing. Bounds and optimality remain numerical. SCIP defines its relative gap using the smaller absolute primal/dual bound in the denominator, so its gap column should not be interpreted as using the same formula as HiGHS. The optional `candidate_callback(candidate, details)` receives independently scored improving feasible incumbents during solving; callback errors interrupt SCIP and propagate to the caller.

The NumPy-only `simulated_annealing_binary_baseline` and `tabu_search_binary_baseline` use the same `ClassicalResult` interface. Both preserve exact cardinality throughout and optimise the original binary objective without a penalty. Each starts from top returns, then diagonal costs, then seeded random subsets. Both accept `seed`, `restarts` (default 16, minimum one), `max_iterations`, and `candidate_callback(candidate, details)`. Callbacks receive independent copies only when the best objective improves, including initial incumbents. Neither method supplies an optimality bound.

```python
from vqe_portfolio.baselines import (
    simulated_annealing_binary_baseline,
    tabu_search_binary_baseline,
)

annealed = simulated_annealing_binary_baseline(
    mu, Sigma, lam=4.0, k=4, seed=0, restarts=16,
    max_iterations=1000, initial_temperature=None, cooling_rate=0.99,
)
tabu = tabu_search_binary_baseline(
    mu, Sigma, lam=4.0, k=4, seed=0, restarts=16,
    max_iterations=200, tabu_tenure=7,
)
```

Annealing proposes 1,000 swaps per restart by default. Its automatic initial temperature is the largest absolute swap cost at the starting portfolio, with a positive numerical floor for a flat objective. Temperature resets at each restart and decreases by `cooling_rate` after each proposal. An explicit `initial_temperature` must be finite and positive; `cooling_rate` must lie strictly between zero and one.

Tabu search takes up to 200 moves per restart by default. Reversing a swap is forbidden for the next `tabu_tenure` iterations (default seven), unless the reversal produces a new best objective across all restarts. Tenure must be a positive integer. A restart stops early if no admissible move remains. Its status is `iteration_limit` if any restart reaches the cap, otherwise `tabu_exhausted`. Annealing reports `iteration_limit` after its scheduled proposals. Both report `trivial` for `k=0` or `k=n`, where there is only one feasible portfolio. Search stopping statuses do not certify global optimality for nontrivial portfolios.

The additional NumPy heuristics use the same `ClassicalResult` interface and optimise the same original binary objective. They accept `seed`, `restarts` (default 16, minimum one), `max_swaps` (default 1,000 per local refinement), and `candidate_callback(candidate, details)`. Callbacks receive independent copies only when the global best objective improves. All crossover, mutation, perturbation, and refined portfolios preserve exactly K assets. GRASP publishes only after construction has selected all K assets.

```python
from vqe_portfolio.baselines import (
    memetic_genetic_binary_baseline,
    iterated_local_search_binary_baseline,
    grasp_binary_baseline,
)

memetic = memetic_genetic_binary_baseline(
    mu, Sigma, lam=4.0, k=4, seed=0, restarts=4,
    population_size=16, generations=25, mutation_rate=0.2,
)
ils = iterated_local_search_binary_baseline(
    mu, Sigma, lam=4.0, k=4, seed=0, max_iterations=100, perturbation_size=2,
)
grasp = grasp_binary_baseline(mu, Sigma, lam=4.0, k=4, seed=0, rcl_alpha=0.3)
```

Memetic search refines every initial member and offspring with improving swaps. Each population starts with top returns, diagonal costs, then random subsets. Parents are chosen through tournaments of size three by default, sampled with replacement. Crossover keeps their common assets and fills remaining positions from their union. Each child mutates by one exchange with probability `mutation_rate`, which must be in [0, 1]. The best population member survives each generation. With population size at least three, one offspring per generation is a random immigrant. Population size must be at least two; duplicates are allowed so small feasible spaces cannot cause endless population-filling retries. `generations` defaults to 25 and `population_size` to 16.

ILS starts from top returns, diagonal costs, then random subsets. Each restart first performs local refinement, then applies 100 perturb-and-refine iterations by default. A perturbation replaces `perturbation_size` distinct selected assets with distinct unselected assets simultaneously (default two, clamped to `min(K, n-K)`). Only improvements beyond the numerical swap tolerance replace the current state; rejected proposals cannot overwrite the best portfolio.

GRASP uses the full objective's marginal insertion costs, including covariance with assets already selected. At each construction step it samples uniformly from costs no larger than `min_cost + rcl_alpha * (max_cost - min_cost)`. `rcl_alpha=0` permits minimum-cost assets, with random ties; `rcl_alpha=1` permits every remaining asset. The default is 0.3. Every completed construction receives swap refinement, and `restarts` controls the number of constructions.

Memetic search reports `generation_limit`, ILS `iteration_limit`, and GRASP `restart_limit` after their configured searches. All three report `trivial` for `K=0` or `K=n`. Local refinement may stop at its `max_swaps` cap. These are heuristic stopping statuses and provide no global optimality bound. To compare them without optional classical dependencies:

```bash
python scripts/generate_comparison_results.py --methods memetic,ils,grasp \
  --asset-counts 8,16,32 --k 4 --seeds 0,1,2 --local-restarts 4 \
  --output results/metaheuristics_summary.csv \
  --trials-output results/metaheuristics_trials.csv
```

Solver interfaces: [SciPy/HiGHS MILP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html), [CVXPY solver options](https://www.cvxpy.org/tutorial/solvers/), [PySCIPOpt quadratic expressions](https://pyscipopt.readthedocs.io/en/stable/tutorials/expressions.html), [SCIP status and gap API](https://pyscipopt.readthedocs.io/en/stable/api/model.html).

`variable_neighborhood_search_binary_baseline` is NumPy-only and preserves exactly K assets. Defaults are `seed=0`, `restarts=16`, `max_iterations=100`, `max_neighborhood=3`, and `max_swaps=1000`. Starts are top returns, diagonal costs, then random subsets. Each restart first refines its start, then cycles through neighbourhood sizes q from one to `min(max_neighborhood, K, n-K)`. A shake exchanges q distinct selected assets with q distinct unselected assets simultaneously, followed by single-swap refinement. Strict improvement beyond the numerical swap tolerance replaces the current state and resets q to one; otherwise q advances, wrapping to one after the largest neighbourhood. `max_iterations` counts shake-and-refine attempts per restart. All limits must be positive integers. Stopping reports `iteration_limit` or `trivial` for K=0/K=n, with no global optimality guarantee.

`projected_gradient_descent_baseline` is a deterministic NumPy-only solver for the separate fractional objective, with nonnegative weights summing to one. It starts from equal weights, computes the full quadratic gradient, and applies Euclidean simplex projection. Covariance eigenvalues below `-1e-10` are rejected. The default step is `1/L`, where `L=2*lambda*largest_eigenvalue(Sigma)`; an optional positive `stepsize` overrides the initial value. Backtracking halves the step, with up to 60 trials per iteration, until quadratic majorisation and non-increase hold within floating-point slack. Zero-curvature objectives use the exact linear simplex solution. Eigenvalue calculation, projection, backtracking, and incumbent publication are included in budgeted execution.

```python
from vqe_portfolio.baselines import (
    variable_neighborhood_search_binary_baseline,
    projected_gradient_descent_baseline,
)

vns = variable_neighborhood_search_binary_baseline(
    mu, Sigma, lam=4.0, k=4, seed=0, max_neighborhood=3,
)
pgd = projected_gradient_descent_baseline(
    mu, Sigma, lam=4.0, max_iterations=10000, tolerance=1e-8,
)
```

PGD reports `converged` when its returned portfolio's numerical first-order gap `gradient@weights - min(gradient)` is at most `tolerance` (default `1e-8`). `iteration_limit` and `line_search_limit` retain the best feasible incumbent; `numerical_limit` means the current iterate met the criterion but the retained best-scoring portfolio's gap remains larger due to numerical score differences. The diagnostic gap is not a certified lower bound or a relative MIP gap. Both `max_iterations` and `tolerance` must be positive, with integer iteration count and finite numerical settings. `lower_bound` and `mip_gap` remain missing. Detailed stopping information appears in the result message, comparison notes, and validation `solver_message`.

Both methods accept `candidate_callback(candidate, details)` and publish independent copies only when the best objective improves, including the initial feasible portfolio. VNS runs per solver seed; PGD runs once per validation/comparison instance. Their binary and fractional objectives remain separate:

```bash
python scripts/generate_comparison_results.py --methods vns,pgd \
  --asset-counts 8,16,32 --k 4 --seeds 0,1,2 --local-restarts 4 \
  --output results/vns_pgd_summary.csv --trials-output results/vns_pgd_trials.csv
```


`frank_wolfe_baseline` supplies a deterministic NumPy-only alternative for the
same fractional objective. It implements forward and away steps with an active
set of simplex vertices and exact bounded quadratic line search. Maximal away
steps remove vertices; sparse iterates do not impose exact cardinality. Defaults
are `max_iterations=10000` and `tolerance=1e-8`. It rejects covariance eigenvalues
below `-1e-10` and supports singular PSD and linear objectives.

```python
from vqe_portfolio.baselines import frank_wolfe_baseline

fw = frank_wolfe_baseline(mu, Sigma, lam=4.0, tolerance=1e-8)
```

Like PGD, it reports `converged`, `iteration_limit`, or `numerical_limit`, retains
copied improving incumbents through `candidate_callback`, and runs once per
comparison or validation instance. The message includes completed forward, away,
and drop step counts, active-set size, and the returned portfolio's numerical
`first_order_gap`. This gap is not independently certified; `lower_bound` and
`mip_gap` remain absent. See the [algorithm guide](docs/algorithms/classical/frank-wolfe.md)
for the update equations and stopping semantics.

```bash
python scripts/generate_comparison_results.py --methods frank-wolfe,pgd \
  --asset-counts 4,8 --output results/fractional_summary.csv \
  --trials-output results/fractional_trials.csv

vqe-portfolio budget --method frank-wolfe --mu "0.1,0.2" \
  --sigma "0.1,0;0,0.2" --lam 0.5 --budget-seconds 10
```


## Shared Problem Definitions and Evaluation

`PortfolioProblem` freezes copies of the returns, covariance, risk aversion, objective family, and cardinality. Its arrays are read-only. All benchmark adapters receive these frozen arrays, and the independent evaluator recomputes feasibility, objective, risk, and return from the returned candidate instead of trusting solver scores. The synthetic, ansatz, and real-data comparison workflows use the same evaluator.

```python
from vqe_portfolio import PortfolioProblem

problem = PortfolioProblem(mu, Sigma, lam=4.0, objective_family="binary_qubo", k=2)
result = problem.evaluate([1, 0, 1], alpha=5.0)  # for a three-asset problem
print(problem.problem_id, result.feasible, result.objective)
```

Use `objective_family="fractional_simplex"` without `k` for continuous allocations. The evaluator requires exact binary entries and cardinality, or nonnegative simplex weights within `1e-8` of the constraints. Wrong shapes, nonfinite values, missing candidates, and constraint violations have no scored objective. Evaluation does not round, project, or normalise candidates. Solver-side cleanup and Top-K fallback remain explicit algorithm behaviour; fallback is recorded in trial notes.

For binary selection, the scored objective is `lam * x.T @ Sigma @ x - mu.T @ x`; risk and return use `x / K`. For fractional allocation, both the objective and descriptive metrics use the returned weights. The existing `binary_qubo` family name is retained for compatibility, but scores exclude penalties. The public `binary_objective(...)` helper still returns penalised QUBO energy for existing callers.

Newly generated CSVs distinguish:

- `problem_id`: SHA-256 of the canonical problem snapshot, including allocation rules and feasibility tolerance. Only compare objective values within the same ID. Penalty strength, solver settings, and seeds are not part of the underlying problem.
- `feasible_rate`: fraction of **returned portfolios** passing independent validation, for every method.
- `sample_feasible_rate`: fraction of sampled binary bitstrings satisfying cardinality, when available. A Top-K fallback can produce a feasible returned portfolio even when no feasible bitstring was sampled.
- `penalized_energy`: trial-level binary energy when a penalty coefficient is supplied; it never controls the benchmark ranking.
- `validation_error`: trial-level reason a candidate was rejected. Invalid trials remain in runtime and feasibility statistics; quality summaries exclude them and are blank if none are valid.

**Schema change:** older committed CSVs and notebook outputs use `feasible_rate` for binary sample feasibility. Those historical values have not been relabelled or regenerated. Use `sample_feasible_rate` for that diagnostic in new outputs; do not mix old and new feasibility columns without accounting for this change.

The synthetic generator writes `<summary-stem>.problems.json` beside its summary CSV, keyed by `problem_id`, containing the exact data for both tracks. Restore an instance with `PortfolioProblem.from_dict(snapshot)` and verify its ID before comparing runs. `problem.to_dict()` provides the same snapshot format for programmatic and real-data workflows. These snapshots define reproducible optimisation inputs; they do not yet implement the full fixed-budget research protocol.

## Small-Instance Validation

The separate validation runner adds multiple independent instances per size, exact-reference objective gaps, target-hit rates, frozen experiment metadata and per-trial checkpoints. It keeps the original example generators unchanged.

```bash
pip install -e ".[classical]"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_small_instance_validation.py \
  --config experiments/small_instance_validation.json \
  --output-dir results/small_instance_validation
```

Use a new output directory to repeat the experiment. To continue an interrupted run, use the same command with `--resume`; configuration, source hashes and recorded environment must match. Already checkpointed trials, including solver errors, are retained. An interruption during a trial leaves that trial to be rerun. There is no concurrent-writer support: run one process per output directory.

See [the experiment design](experiments/README.md) and [the saved pilot report](results/small_instance_validation/report.md). This validates small-instance behaviour on a noiseless simulator; it does not implement matched end-to-end time budgets or establish quantum advantage.

### Before and After Training

Measure the exact single-draw probability of a feasible, target-quality portfolio at the initial and trained parameters of Binary VQE, QAOA X and QAOA XY:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_training_diagnostic.py \
  --pilot-manifest results/small_instance_validation/manifest.json \
  --output-dir results/my_training_diagnostic
```

The runner inherits settings and frozen inputs from the validation manifest. It writes paired metrics, initial/final parameters, exact distributions, a plot and a report. Add `--resume` to continue a matching interrupted run. See [the diagnostic design](experiments/README.md#before-and-after-training-diagnostic) and [saved report](results/training_diagnostic/report.md). This evaluates binary circuit distributions without sampling or Top-K repair; Variational quantum allocator is outside this diagnostic.

### Sampling Allowances

Calculate the chance of at least one target-quality portfolio in 16, 64 or 512 independent draws from the saved initial/trained circuit distributions, alongside uniform feasible classical sampling:

```bash
python scripts/run_sampling_allowance.py \
  --diagnostic-dir results/training_diagnostic \
  --draws 16,64,512 \
  --output-dir results/my_sampling_allowance
```

This analysis uses the completed checkpoint without retraining or drawing samples. It validates the raw probability vectors, transforms each run before aggregation, preserves source failures and records input hashes. Use a new output directory to repeat it. See [the method and assumptions](experiments/README.md#sampling-allowance-diagnostic) and [saved report](results/sampling_allowance/report.md). Draw counts measure sampling reliability; execution and training costs are excluded.

### Training Effort

Compare 0, 25 and 100 optimisation steps along a single Adam trajectory per instance/method/seed, preserving optimiser state:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_training_effort.py \
  --diagnostic-dir results/training_diagnostic \
  --steps 0,25,100 --draws 16,64,512 \
  --output-dir results/my_training_effort
```

The runner inherits the earlier diagnostic settings, checks the initial/25-step states against saved evidence, and records exact probabilities, sampling reliability and cumulative training times. Add `--resume` to skip completed trajectories; an interrupted trajectory restarts from its seed to preserve Adam's full state. See [the study design](experiments/README.md#training-effort-development-study) and [saved report and paired plots](results/training_effort/report.md). This is development sensitivity analysis, not a held-out or matched-budget advantage test.

### Reachability, Depth and Penalty Diagnostics

Investigate the eight-security failures using exact depth-1 XY support, higher-depth support bounds and controlled training variants:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_failure_diagnostics.py \
  --effort-dir results/training_effort \
  --xy-depths 1,2,3 --vqe-depths 2,3 --penalties 0.05,0.5,5 \
  --output-dir results/my_failure_diagnostics
```

The runner preserves the frozen problem objectives and reuses verified baseline trajectories. New variants inherit 100-step effort and seeds. It saves exact amplitude polynomials, target-support comparisons, angle sensitivity, penalty sufficiency checks, VQE modal/gradient/swap diagnostics and figures. Use `--resume` to skip completed variants. See [the assumptions and reproduction details](experiments/README.md#eight-security-failure-diagnostics) and [saved report](results/failure_diagnostics/report.md). Higher-depth support bounds and illustrative swap barriers are not proofs of successful optimisation.

## Solver reliability controls (v0.3.1)

The default circuits, penalties, and single-start training remain available for
comparison with historical experiments. New settings are explicit algorithm
choices and must be frozen before evaluation.

For XY QAOA, `xy_initial_state="dicke"` starts in a uniform superposition of all
exact-K portfolios. This lets the first cost angle affect relative phases,
whereas the legacy `"basis"` initial state makes that angle a global phase.
`xy_topology="complete"` exchanges every pair of assets; `"ring"` retains the
historical ordered even/odd/wrap edges. Both preserve cardinality. These settings
also apply to QAOA lambda sweeps.

```python
from vqe_portfolio import QAOAConfig, run_qaoa

config = QAOAConfig(k=2, p=1, mixer="xy", xy_initial_state="dicke",
                    xy_topology="complete", steps=25, seed=0)
result = run_qaoa(mu, sigma, config)
```

The Dicke implementation uses `StatePrep` with an explicit `2**n` amplitude vector.
It supports small simulator experiments; hardware decomposition and its cost
must be evaluated separately. Complete networks have `O(n**2)` exchange edges.
Neither option guarantees better trained portfolios.

Binary VQE adds these controls:

| Setting | Behavior |
| --- | --- |
| `alpha="auto"` | Reference-free conservative sufficient penalty derived from a feasible upper score and a coefficient-wise lower bound |
| `normalize_energy=True` | Divide training energy by the nonidentity Hamiltonian coefficient L1 norm; report traces in original energy units |
| `restarts=3` | Train three initializations and select by observed penalized training energy |
| `keep_best=True` | Return the best observed training-cost iterate, including initialization |
| `stepsize_decay=0.03` | Use `stepsize / (1 + 0.03*t)` at zero-based update `t` |

`steps` is **per restart**, so total updates are `steps * restarts`. Best-iterate
selection and restart selection use the configured training cost, never a reference optimum
or target-quality score. On finite-shot training these are noisy observations.
`effective_alpha`, `energy_scale`, and `restart_energies` record the resolved
settings and restart scores. The energy trace describes the selected restart's
trajectory, which can end after the returned best iterate. Auto penalties also
work for QAOA, but a sufficient penalty guarantees only that the global
penalized optimum is feasible; it does not guarantee optimizer convergence.

For the auto penalty, write the original binary objective as linear coefficients
`a_i` and pair coefficients `b_ij`. The lower bound is
`L = sum(min(a_i,0)) + sum(min(b_ij,0))`; `U` is the score of a feasible top-return
portfolio. Choosing `alpha > max(0,U-L)` is sufficient because every infeasible
bitstring incurs at least one unit of squared cardinality violation. No
exhaustive reference is used. This bound can be conservative, which motivates
comparing normalization, penalties, and restart settings experimentally.

```bash
vqe-portfolio binary --mu "0.1,0.2,0.15" \
  --sigma "0.1,0,0;0,0.2,0;0,0,0.1" --k 2 --lam 4 \
  --alpha auto --normalize-energy --restarts 3 --keep-best --stepsize-decay 0.03

vqe-portfolio qaoa --mu "0.1,0.2,0.15" \
  --sigma "0.1,0,0;0,0.2,0;0,0,0.1" --k 2 --lam 4 --mixer xy \
  --xy-initial-state dicke --xy-topology complete
```

Optimizer traces now report **post-update** costs at the logged parameter
snapshot, including the final step even when it is not a logging multiple.
Logging requires an additional cost evaluation. This changes runtime and can
change finite-shot random streams; historical traces and timings are preserved.
Use new output directories for studies after upgrading.

## CVaR training

Binary VQE and QAOA accept `training_objective="cvar"` and
`cvar_alpha=0.25` in their Python configs, or `--training-objective cvar
--cvar-alpha 0.25` on their CLI commands, including market-data commands.
The default remains expectation training. The tail fraction must be finite
and in `(0, 1]`.

CVaR averages the best-cost fraction of circuit outcomes, including fractional
weight at the quantile boundary and penalties on infeasible bitstrings. It is
an optimization training objective, not financial downside-risk CVaR.
Analytic training uses differentiable probabilities and Adam; finite-shot
training with a tail below 1 uses SPSA. A tail fraction of 1 follows the
existing expectation/Adam path. CVaR also applies to lambda sweeps, Binary
VQE restart and best-iterate selection, and budgeted execution.

Traces and `restart_energies` report the configured cost in original units.
Results include `training_objective`, `cvar_alpha`, and `training_optimizer`.
See [CVaR training](docs/algorithms/quantum/cvar.md) for examples, SPSA schedules,
shot costs and simulator limits.

## End-to-end execution budgets (v0.3.1)

`run_budgeted(problem, method, budget_seconds, config=..., seed=...)` executes
Binary VQE, QAOA X/XY, Variational quantum allocator, HiGHS, SCIP, OSQP, QP rounding, swap search, simulated
annealing, tabu search, memetic search, ILS, GRASP, VNS, PGD, or uniform exact-K sampling. An explicit quantum configuration supplies its own seed; the `seed` argument
applies to classical methods and automatically constructed quantum configurations.
It returns a common `BudgetResult` containing the best feasible
candidate, its independently recomputed original objective, timestamped
incumbents, termination status, native solver status, error, and runtime.

```python
from vqe_portfolio import PortfolioProblem, BinaryVQEConfig, run_budgeted

# Put spawned execution behind this guard when running a Python script.
if __name__ == "__main__":
    problem = PortfolioProblem([0.1, 0.2], [[0.1, 0], [0, 0.2]],
                               4.0, "binary_qubo", 1)
    config = BinaryVQEConfig(k=1, lam=4.0, steps=10000, log_every=5,
                             alpha="auto", shots_sample=64)
    result = run_budgeted(problem, "binary-vqe", 10.0, config=config)
    print(result.status, result.candidate, result.objective)
```

Runs use an isolated **spawned process**. The deadline starts before worker
startup and includes imports, process communication, encoding, device setup,
optimization, candidate measurement, sampling, repair, evaluation, and delivery.
Market-data acquisition and preparation of a shared `PortfolioProblem` precede
timing. Only independently feasible portfolios received by the parent before
the deadline are eligible. A native solver call can be terminated even when it
does not support callbacks. Pending native results are unavailable until they
return; HiGHS/OSQP currently publish their final candidate only. SCIP, simulated
annealing, tabu search, memetic search, ILS, and GRASP publish improving feasible incumbents. Swap search
publishes feasible starts and improving swaps; uniform sampling publishes
improving exact-K samples throughout its allowance.

`qp-round` publishes improving binary incumbents after its relaxation returns,
including the rounded selection and any improving swaps. Relaxation solving,
rounding, refinement, and publication all consume the common deadline.

Quantum methods publish candidates every `log_every` updates and at the end.
Binary candidates come from the best feasible sample, with measured-marginal
Top-K fallback if none is sampled. Fractional candidates use measured allocations.
These evaluations consume the allowance and may change finite-shot streams, so
freeze checkpoint cadence and shot counts as algorithm settings. The incumbent
is ranked by the original feasible-portfolio objective, independently of energy
or expected feasibility. Ordinary solver functions also accept an optional
`candidate_callback(candidate, details)` for integrations without process budgets.

`status` is `completed`, `time_limit`, or `error`; an error or timeout can retain
an earlier feasible incumbent. A run with `candidate=None` is a failed return.
`elapsed_seconds` includes worker shutdown; `overrun_seconds` explicitly reports
cleanup beyond the allowance and never extends candidate eligibility. Process
startup can consume an entire short allowance. This API does not assume a
persistent worker, amortized startup, or QPU queue behavior, and requires a
spawn-compatible Python entry point rather than an interactive REPL.

The existing solver/data commands accept `--budget-seconds`. The `budget`
subcommand also exposes classical methods, with a default allowance of 10 seconds:

```bash
vqe-portfolio budget --method swap --budget-seconds 10 --lam 4 --k 2 \
  --mu "0.1,0.2,0.15" --sigma "0.1,0,0;0,0.2,0;0,0,0.1"
```

Use `--method anneal`, `tabu`, `memetic`, `ils`, `grasp`, `vns`, or `qp-round` for these binary heuristics, `--method scip` for the quadratic integer solver, or `--method pgd` or `--method frank-wolfe` for fractional allocation. Each publishes improving incumbents during optimisation so a timeout can retain the best portfolio received before the deadline. The budget interface uses their default settings and may finish before the allowance when configured limits or convergence criteria are reached. PGD and Frank–Wolfe do not require `--k`.

Reproduce fixed-budget runs with frozen references and resumable manifests:

```bash
python scripts/run_budget_benchmark.py --budgets 1,10,60
python scripts/run_budget_benchmark.py --budgets 1,10,60 --resume
python scripts/run_solver_reliability.py
```

To compare just these heuristics under fixed deadlines, use a fresh output directory:

```bash
python scripts/run_budget_benchmark.py --methods anneal,tabu \
  --budgets 1,10,60 --output-dir results/heuristics_budget
```

The benchmark saves every failed run, incumbent trajectory, settings, reference
kind, source hashes, and environment. `reliability.csv` reports empirical T90
per instance, including censored failures; blank T90 means fewer than 90% reach
the target within that allowance. Numerical fractional references retain their
numerical status. These small development pilots do not satisfy the held-out
sample sizes and uncertainty analysis in the full research protocol.

### Fresh-instance comparison and timing diagnostics

Run the frozen fresh-instance configuration in a new output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_budget_benchmark.py \
  --config experiments/fresh_instance_budget.json \
  --output-dir results/fresh_instance_budget
```

Add `--resume` to continue with matching sources, settings, and environment.
CLI arguments override JSON defaults. `--variants` selects named presets and/or
ordinary method names; it takes precedence over `--methods` when nonempty.
Named presets inherit the v0.3.1 reliability-study settings, independently of
the generic solver flags: `binary-legacy`, `binary-improved`, `xy-basis-ring`,
`xy-dicke-ring-trained`, and `xy-dicke-ring-untrained`. The improved Binary VQE
preset divides the declared total update count among three restarts; its total
must be a positive multiple of three. Trial rows save every resolved solver
configuration.

The configuration uses fresh factor-model instance seeds 707, 808, and 909 at
4, 6, and 8 assets, two solver seeds, and independent 1/10/60-second runs. It
compares the presets with QAOA X, HiGHS, swap search, uniform feasible sampling,
and a separate Variational quantum allocator/OSQP control track. `--exclude-manifests` freezes
the hashes and problem IDs of prior development studies; overlap is rejected
before any solver trial. Settings are fixed before evaluating the new inputs.
This is a small fresh-instance check, not the protocol's full held-out study.

The untrained XY variant repeatedly executes finite-shot batches at the same
production random parameters used to initialize its trained counterpart. It
uses zero Adam updates and no exact probability vector. Both have the same
startup-inclusive allowance and 64-shot batch size; the untrained control can
spend more of that allowance sampling. The trained Dicke-ring variant publishes
candidates during its 24 updates, then repeatedly samples the final trained
parameters for the remaining time. Other trained presets retain their finite
update caps and ordinary final sampling. Uniform sampling and both Dicke-ring
sampling loops stop early if they have actually
sampled all `binomial(n, K)` feasible portfolios. This stopping condition uses
only the feasible-domain size, with no reference target; the best portfolio
cannot improve after full coverage. Otherwise sampling continues to the deadline.

`run_budgeted` results now include `timings`: worker phase durations for startup
(spawn/imports), setup, optimization, and candidate processing; mean delivery
delay among accepted incumbents; total parent validation time; and cleanup time.
The last observed active phase is charged through a timeout and identified by
`interrupted_phase`. Startup censored at the deadline is not a completed startup
measurement. Delivery and parent validation overlap worker work, so they must
not be added to the phase totals. Parent validation must complete within the
allowance for a candidate to count. Logging these diagnostics consumes time too.
Phases follow execution boundaries: lazy circuit compilation belongs to the
first phase executing that circuit, and a classical baseline's optimization
phase includes model construction and optional imports inside its solver call.
These are end-to-end implementation diagnostics, not native kernel timings.

The runner exports `timings.csv`, descriptive trained/comparator pairs in
`pairs.csv`, and the existing raw trials, summaries, incumbent histories, and T90
records. Paired quality differences require two feasible returns; failure flags
and target-hit outcomes remain explicit. Historical reports are preserved.
