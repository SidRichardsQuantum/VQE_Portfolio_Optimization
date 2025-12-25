# Portfolio Optimization via VQE

This repository implements **portfolio optimization using Variational Quantum Eigensolvers (VQE)** in a clean, modular **Python package** with lightweight notebook clients.

Two complementary quantum formulations are provided:

- **Binary VQE** — asset *selection* under a cardinality constraint (QUBO → Ising → VQE)
- **Fractional VQE** — long-only *allocation* on the simplex using a constraint-preserving quantum parameterization

All quantum logic lives in `src/vqe_portfolio/`; most notebooks act purely as **clients** for running experiments, generating plots, and reproducing results.

---

## 🚀 Implemented Methods

### 1️⃣ Binary VQE (Asset Selection)

Select exactly **K assets** by solving a constrained mean–variance problem:

$$
\min_{x \in \{0,1\}^n}
\;\lambda\, x^\top \Sigma x
\;-\;\mu^\top x
\;+\;\alpha(\mathbf{1}^\top x - K)^2
$$

- QUBO formulation mapped to an **Ising Hamiltonian**
- Hardware-efficient **RY + CZ ring** ansatz
- VQE minimizes ⟨H⟩
- Outputs:
  - Inclusion probabilities
  - Sampled portfolios
  - Top-K projections
  - λ-sweep and efficient frontier

Notebook client:
- `notebooks/Binary_VQE.ipynb`

---

### 2️⃣ Fractional VQE (Continuous Allocation)

Solve the long-only mean–variance problem on the simplex:

$$
\min_{w \in \Delta}
\;-\mu^\top w + \lambda\, w^\top \Sigma w
\quad\text{with}\quad
\Delta=\{w\ge0,\sum_i w_i=1\}
$$

- Simplex enforced **by construction**
- Circuit readout → weights via
  $$
  w_i = \frac{(1-\langle Z_i\rangle)/2}{\sum_j (1-\langle Z_j\rangle)/2}
  $$
- No penalty tuning required
- Warm-started λ sweeps
- Efficient frontier computed from allocations

Notebook client:
- `notebooks/Fractional_VQE.ipynb`

---

## 🗂 Repository Structure

```text
src/
└── vqe_portfolio/
    ├── binary.py        # Binary VQE (QUBO / Ising formulation)
    ├── fractional.py    # Fractional VQE (simplex parameterization)
    ├── frontier.py      # Efficient frontier utilities
    ├── ansatz.py        # Shared circuit ansätze
    ├── optimize.py      # Optimizer loops
    ├── metrics.py       # Risk / return utilities
    ├── plotting.py      # Centralized plotting helpers
    ├── data.py          # Market data utilities
    └── types.py         # Dataclasses for configs & results

notebooks/
├── Binary_VQE.ipynb
├── Fractional_VQE.ipynb
└── images/              # Auto-generated figures
```


---

## ▶️ Running the Examples

### Install dependencies

```bash
pip install -r requirements.txt
```

or editable:
```bash
pip install -e .
```

### Run notebooks
Open and execute:
- `notebooks/Binary_VQE.ipynb`
- `notebooks/Fractional_VQE.ipynb`

All figures are generated automatically in `notebooks/images/`.

---

## 📚 Documentation

- **Theory & derivations**: [`THEORY.md`](THEORY.md)
- **Results & figures**: [`RESULTS.md`](RESULTS.md)

The theory document derives:
- QUBO → Ising mappings
- Constraint handling
- Quantum measurement → portfolio interpretation

The results document summarizes:
- Convergence behavior
- λ sweeps
- Efficient frontiers

---

## 🧠 Why This Matters

This project demonstrates:

- Translating **financial optimization problems** into quantum Hamiltonians
- Careful constraint handling (cardinality vs simplex)
- Clean separation of **research logic** and **experimental notebooks**
- Reproducible hybrid quantum–classical workflows
- Production-ready Python packaging for quantum algorithms

The architecture is intentionally extensible to:
- Alternative ansätze
- Noise models
- Classical baselines (e.g. Markowitz)
- Other QUBO-style optimization problems

---

## 🧾 References

- QUBO overview: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization
- PennyLane documentation: https://docs.pennylane.ai

---

**Author**: Sid Richards  
GitHub: [@SidRichardsQuantum](https://github.com/SidRichardsQuantum)  
LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

MIT License — see [LICENSE](LICENSE)
