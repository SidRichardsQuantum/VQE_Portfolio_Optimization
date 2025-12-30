# Usage Guide — vqe‑portfolio

This document shows how to use the `vqe_portfolio` package **without notebooks**, directly from Python.
Notebooks in this repository are thin clients around the same API.

---

## 🖥 Command-Line Interface (CLI)

The package provides a first-class CLI for running VQE portfolio optimization **without writing Python code**.

After installation, the CLI is available as:

```bash
vqe-portfolio --help
````

You can also invoke it via:

```bash
python -m vqe_portfolio --help
```

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
  --steps 80 \
  --out binary_result.json
```

This produces a JSON file containing:

* optimized circuit parameters
* inclusion probabilities
* Top-K selection
* sampled bitstrings and counts
* optimization trace

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
  --steps 100 \
  --out fractional_result.json
```

The output JSON includes:

* optimized circuit parameters
* portfolio weights
* cost trace

### Input formats

You may provide inputs in either form:

**Inline**

```bash
--mu "0.1,0.2,0.3"
--sigma "1,0.1;0.1,2"
```

**JSON**

```json
{
  "mu": [...],
  "sigma": [...]
}
```

JSON is recommended for reproducibility and larger problem sizes.

### Reproducibility via CLI

All CLI commands respect the same reproducibility controls as the Python API:

```bash
--seed 0
```

### Relationship to the Python API

The CLI is a **thin client** over the same public API used by notebooks and scripts:

* `vqe-portfolio binary` → `run_binary_vqe`
* `vqe-portfolio fractional` → `run_fractional_vqe`

No logic is duplicated.

---

## 1️⃣ Minimal Example (Synthetic Data)

```python
import numpy as np
from vqe_portfolio import run_fractional_vqe, FractionalVQEConfig

# Synthetic mean returns and covariance
mu = np.array([0.10, 0.12, 0.07])
Sigma = np.array([
    [0.04, 0.01, 0.00],
    [0.01, 0.09, 0.02],
    [0.00, 0.02, 0.03],
])

cfg = FractionalVQEConfig(
    lam=5.0,
    steps=100,
    stepsize=0.25,
    shots=None,
)

res = run_fractional_vqe(mu, Sigma, cfg)
print(res.weights)
```

This solves the long‑only mean–variance problem on the simplex **by construction**.

---

## 2️⃣ Binary VQE (Asset Selection)

```python
import numpy as np
from vqe_portfolio import run_binary_vqe, BinaryVQEConfig

mu = np.array([0.10, 0.12, 0.07, 0.09])
Sigma = 0.02 * np.eye(4)

cfg = BinaryVQEConfig(
    k=2,
    lam=4.0,
    alpha=2.0,
    steps=80,
)

res = run_binary_vqe(mu, Sigma, cfg)
print("Top‑K selection:", res.x_topk)
print("Probabilities:", res.x_prob)
```

This selects exactly `k` assets using a QUBO → Ising → VQE formulation.

---

## 3️⃣ λ‑Sweeps and Efficient Frontiers

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
    mu, Sigma, res.lambdas, res.allocs_by_lambda
)
```

The resulting `Frontier` object contains risk, return, λ, and portfolio weights.

---

## 4️⃣ Real Market Data

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

cfg = FractionalVQEConfig(lam=5.0, steps=100)
res = run_fractional_vqe(mu.values, Sigma.values, cfg)
print(res.weights)
```

---

## 5️⃣ Reproducibility

```python
from vqe_portfolio.utils import set_global_seed
set_global_seed(0)
```

All optimization loops and random initializations respect the global seed.

---

## 6️⃣ Notebooks as Clients

All notebooks in `notebooks/` simply:

* import the public API
* call `run_binary_vqe` / `run_fractional_vqe`
* generate plots via `vqe_portfolio.plotting`

They contain **no core logic**.

---

## 7️⃣ What This Package Is (and Is Not)

**This package is:**

* A research‑grade quantum optimization toolkit
* Deterministic, testable, and CI‑validated
* Designed for experimentation and extension

**This package is not:**

* A production trading system
* A performance‑optimized classical solver
* A claim of quantum advantage

---

For theory, see `THEORY.md`.
For experimental results, see `RESULTS.md`.

---

📘 **Author**: Sid Richards  
MIT License — see [LICENSE](LICENSE)
