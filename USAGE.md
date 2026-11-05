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

### Fractional VQE (Continuous Allocation)

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
vqe-portfolio fractional \
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

Supported Fractional VQE ansatz values are `ry`, `ry_cz`, and `ry_rz_cz`.

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
- `vqe-portfolio fractional` → `run_fractional_vqe`
- benchmark notebooks/scripts → `vqe_portfolio.research.comparison` workflows and `vqe_portfolio.benchmark` evaluation helpers and public ansatz config options

No logic is duplicated.

---

## 1. Minimal Example (Synthetic Data)

```python
import numpy as np
from vqe_portfolio import run_fractional_vqe, FractionalVQEConfig

mu = np.array([0.10, 0.12, 0.07])

Sigma = np.array([
    [0.04, 0.01, 0.00],
    [0.01, 0.09, 0.02],
    [0.00, 0.02, 0.03],
])

cfg = FractionalVQEConfig(
    lam=5.0,
    ansatz="ry",
    depth=1,
    steps=100,
    stepsize=0.25,
    shots=None,
)

res = run_fractional_vqe(mu, Sigma, cfg)

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
from vqe_portfolio import fractional_lambda_sweep
from vqe_portfolio.frontier import fractional_frontier_from_allocs
from vqe_portfolio.types import LambdaSweepConfig

cfg = FractionalVQEConfig(lam=1.0, steps=80)

sweep = LambdaSweepConfig(
    lambdas=[0.5, 1.0, 2.0, 5.0, 10.0],
    steps_per_lambda=60,
    warm_start=True,
)

res = fractional_lambda_sweep(mu, Sigma, cfg, sweep)

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
from vqe_portfolio import get_stock_data, run_fractional_vqe
from vqe_portfolio.types import FractionalVQEConfig

mu, Sigma, prices = get_stock_data(
    ["AAPL", "MSFT", "GOOGL", "AMZN"],
    start="2024-01-01",
    end="2025-01-01",
    shrink="lw",
)

cfg = FractionalVQEConfig(
    lam=5.0,
    steps=100,
)

res = run_fractional_vqe(
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
- call `run_binary_vqe`, `run_qaoa`, or `run_fractional_vqe`
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
| Classical multistart swap search | Exact-K selection, best-improvement swaps from top-return, diagonal-cost, and random starts | Seeded heuristic with no global optimality guarantee |

The HiGHS formulation introduces a continuous variable `y_ij` per pair, constrained by `y_ij <= x_i`, `y_ij <= x_j`, and `y_ij >= x_i + x_j - 1`, with all variables bounded in `[0, 1]` and `x` binary. This represents both positive and negative covariance couplings exactly. Its model size grows quadratically; it is a stronger general-purpose comparator, not a claim to be the best possible classical solver.

`--solver-time-limit` limits the HiGHS solve only. Reported runtime includes the entire solver helper call, including model construction and candidate cleanup; it can exceed that limit. OSQP uses accuracy tolerances, and swap search uses restart/swap limits, not that time limit. These runs do **not** yet implement the fixed end-to-end budgets in the quantum-advantage protocol.

`--seeds` controls independent multistart-search runs as well as quantum runs. Each local-search run uses `--local-restarts` starts (minimum two). `--k` fixes cardinality across requested sizes. Defaults remain compatible with the existing examples: `all` selects the original methods and excludes optional solvers. Use `--methods all,classical-solvers` to include the new methods alongside the original small-instance references and quantum methods. Avoid `classical`/`all` for large instances because the old exact references enumerate subsets.

The CSVs add `solver_status`, `lower_bound`, and `mip_gap`; legacy methods leave them blank. Trial rows retain unsuccessful solver returns with blank objective/portfolio values and zero feasibility. Summary quality statistics include feasible returns only, while runtime, trial count, and feasibility include failures. Inspect trial rows before drawing comparisons. A reported bound is the numerical solver bound, not an independent exact-arithmetic certificate. Missing dependencies or solver exceptions stop the run visibly rather than silently substituting a weaker solver.

Programmatic entry points are `convex_markowitz_baseline`, `mixed_integer_binary_baseline`, and `local_search_binary_baseline` in `vqe_portfolio.baselines`. They return a `ClassicalResult` with the candidate, recomputed original objective, termination status, and available bound metadata. The local-search helper also accepts `max_swaps` (default 1,000 per restart).

Solver interfaces: [SciPy/HiGHS MILP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html), [CVXPY solver options](https://www.cvxpy.org/tutorial/solvers/).


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

The runner inherits settings and frozen inputs from the validation manifest. It writes paired metrics, initial/final parameters, exact distributions, a plot and a report. Add `--resume` to continue a matching interrupted run. See [the diagnostic design](experiments/README.md#before-and-after-training-diagnostic) and [saved report](results/training_diagnostic/report.md). This evaluates binary circuit distributions without sampling or Top-K repair; Fractional VQE is outside this diagnostic.

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
selection and restart selection use training energy, never a reference optimum
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

## End-to-end execution budgets (v0.3.1)

`run_budgeted(problem, method, budget_seconds, config=..., seed=...)` executes
Binary VQE, QAOA X/XY, Fractional VQE, HiGHS, OSQP, swap search, or uniform exact-K
sampling. An explicit quantum configuration supplies its own seed; the `seed` argument
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
return; HiGHS/OSQP currently publish their final candidate only. Swap search
publishes feasible starts and improving swaps; uniform sampling publishes
improving exact-K samples throughout its allowance.

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

Reproduce fixed-budget runs with frozen references and resumable manifests:

```bash
python scripts/run_budget_benchmark.py --budgets 1,10,60
python scripts/run_budget_benchmark.py --budgets 1,10,60 --resume
python scripts/run_solver_reliability.py
```

The benchmark saves every failed run, incumbent trajectory, settings, reference
kind, source hashes, and environment. `reliability.csv` reports empirical T90
per instance, including censored failures; blank T90 means fewer than 90% reach
the target within that allowance. Numerical fractional references retain their
numerical status. These small development pilots do not satisfy the held-out
sample sizes and uncertainty analysis in the full research protocol.
