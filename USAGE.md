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
- benchmark notebooks/scripts → `vqe_portfolio.comparison` helpers and public ansatz config options

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

Programmatic entry points are `convex_markowitz_baseline`, `mixed_integer_binary_baseline`, and `local_search_binary_baseline` in `vqe_portfolio.classical`. They return a `ClassicalResult` with the candidate, recomputed original objective, termination status, and available bound metadata. The local-search helper also accepts `max_swaps` (default 1,000 per restart).

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
