# Portfolio Optimization via VQE

[![PyPI](https://img.shields.io/pypi/v/vqe-portfolio.svg)](https://pypi.org/project/vqe-portfolio/)
[![Python](https://img.shields.io/pypi/pyversions/vqe-portfolio.svg)](https://pypi.org/project/vqe-portfolio/)
[![License](https://img.shields.io/pypi/l/vqe-portfolio.svg)](LICENSE)
[![CI](https://github.com/SidRichardsQuantum/VQE_Portfolio_Optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/SidRichardsQuantum/VQE_Portfolio_Optimization/actions)

This package implements **portfolio optimization using Variational Quantum Eigensolvers (VQE)** as a clean, testable, and reusable **Python library**, with notebooks acting purely as *clients*.

Three complementary quantum formulations are provided:

- **Binary VQE** — asset *selection* under a cardinality constraint (QUBO → Ising → VQE)
- **QAOA** — gate-based combinatorial optimization using alternating cost and mixer Hamiltonians
- **Fractional VQE** — long-only *allocation* on the simplex using a constraint-preserving quantum parameterization

All core logic lives in `src/vqe_portfolio/`; notebooks and examples simply call the public API.

---

## 🚀 Implemented Methods

### 1. Binary VQE (Asset Selection)

Select exactly **K assets** by solving a constrained mean–variance problem:

$$
\min_{x \in \{0,1\}^n}
\;\lambda\, x^\top \Sigma x
\;-\;\mu^\top x
\;+\;\alpha(\mathbf{1}^\top x - K)^2
$$

**Highlights**

- QUBO formulation mapped to an **Ising Hamiltonian**
- Hardware-efficient **RY + CZ ring** ansatz
- VQE minimizes ⟨H⟩ directly
- Outputs include probabilities, samples, Top‑K projections, λ‑sweeps, and efficient frontiers

Notebook client:

- `notebooks/Binary.ipynb`
- `notebooks/examples/02_Real_Example.ipynb`

---

### 2. QAOA (Binary Asset Selection)

Solve the same constrained mean–variance problem using the **Quantum Approximate Optimization Algorithm (QAOA)**:

$$
\min_{x \in \{0,1\}^n}
\;\lambda\, x^\top \Sigma x
\;-\;\mu^\top x
\;+\;\alpha(\mathbf{1}^\top x - K)^2
$$

**Highlights**

- Uses the same QUBO → Ising mapping as Binary VQE
- Alternating operator ansatz:
  - cost unitary $e^{-i\gamma H_C}$
  - mixer unitary $e^{-i\beta H_M}$
- Supports:
  - standard **X mixer**
  - **XY mixer** for improved constraint structure
- Produces:
  - bitstring samples
  - marginal selection probabilities
  - Top-K projections
  - feasible candidate solutions
  - λ-sweeps

Notebook client:

- `notebooks/QAOA.ipynb`
- `notebooks/examples/03_Real_Example.ipynb`

---

### 3. Fractional VQE (Continuous Allocation)

Solve the long-only mean–variance problem on the simplex:

$$
\min_{w \in \Delta}\; -\mu^\top w + \lambda\, w^\top \Sigma w
\quad\text{with}\quad
\Delta={w\ge0,\sum_i w_i=1}
$$

**Highlights**

- Simplex constraint enforced **by construction**
- No penalty tuning required
- Smooth λ‑sweeps with optional warm starts
- Efficient frontier computed from allocations

Notebook clients:

- `notebooks/Fractional.ipynb`
- `notebooks/examples/01_Real_example.ipynb`

---

## 🧠 Why Quantum Here?

Classical mean–variance portfolio optimization is well understood and efficiently solvable *in its simplest form*.
However, many practically relevant extensions introduce **combinatorial structure** that scales poorly with problem size.

This project focuses on those regimes.

### What is classically easy
- Unconstrained or long-only Markowitz optimization
- Convex quadratic objectives on the simplex
- Small-scale cardinality constraints via heuristics

### What becomes hard
- **Exact cardinality constraints** (select exactly *K* assets)
- Discrete–continuous hybrid decision spaces
- Exhaustive exploration of correlated asset subsets
- Non-convex penalty landscapes introduced by constraints

These settings naturally map to **QUBO / Ising formulations**, which are native to near-term quantum algorithms such as **VQE** and **QAOA**.

### Why VQE is a natural research tool
- VQE directly minimizes ⟨H⟩ for problem-encoded Hamiltonians
- Constraints can be enforced **structurally** (fractional case) or via penalties (binary case)
- Hybrid quantum–classical loops align with existing optimization workflows
- The framework cleanly supports:
  - Ansatz experimentation
  - Noise and shot studies
  - Warm-started parameter sweeps

### What this project does *not* claim
- Quantum advantage over classical solvers
- Near-term production readiness
- Superiority to specialized classical optimizers

Instead, this repository provides a **carefully engineered research baseline** for exploring how constrained financial optimization problems behave when expressed in quantum-native representations.

---

## 📦 Installation

Base install (quantum algorithms only):

```bash
pip install vqe-portfolio
```

With real market data utilities:

```bash
pip install "vqe-portfolio[data]"
```

With classical Markowitz baseline:

```bash
pip install "vqe-portfolio[markowitz]"
```

For development:

```bash
pip install -e ".[dev]"
```

---

## 🗂 Repository Structure

```
src/
└── vqe_portfolio/
    ├── binary.py        # Binary VQE (QUBO / Ising formulation)
    ├── qaoa.py          # QAOA portfolio optimization
    ├── fractional.py    # Fractional VQE (simplex parameterization)
    ├── frontier.py      # Efficient frontier utilities
    ├── ansatz.py        # Shared circuit ansätze
    ├── optimize.py      # Optimizer loops
    ├── metrics.py       # Risk / return utilities
    ├── plotting.py      # Centralized plotting helpers
    ├── data.py          # Market data utilities
    └── types.py         # Dataclasses for configs & results

notebooks/
├── Binary.ipynb
├── QAOA.ipynb
├── Fractional.ipynb
├── examples/
│   ├── 01_Real_example.ipynb
│   ├── 02_Real_Example.ipynb
│   └── 03_Real_Example.ipynb
└── images/
```

---

## 📖 Usage

This package can be used **both programmatically (Python API)** and **from the command line (CLI)**.

See **[USAGE.md](USAGE.md)** for:

- Command-line interface (CLI) usage
- Minimal API examples
- Synthetic-data quickstart
- Real-data workflows
- λ-sweeps and efficient frontiers

---

## 📚 Additional Documentation

- **Theory & derivations**: [`THEORY.md`](THEORY.md)

---

## 🧠 Why This Matters

This project demonstrates:

- Mapping **financial optimization problems** to quantum Hamiltonians
- Clean constraint handling (cardinality vs simplex)
- A strict separation between **research code** and **experiment clients**
- Reproducible hybrid quantum–classical workflows
- Production‑grade packaging and CI for quantum algorithms

---

## 🧾 References

- QUBO overview: [https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization)
- PennyLane documentation: [https://docs.pennylane.ai](https://docs.pennylane.ai)

---

## Author

**Sid Richards**

LinkedIn:
[https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

GitHub:
[https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

---

## License

MIT License — see [LICENSE](LICENSE)
