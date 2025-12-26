# Portfolio Optimization via VQE

This package implements **portfolio optimization using Variational Quantum Eigensolvers (VQE)** as a clean, testable, and reusable **Python library**, with notebooks acting purely as *clients*.

Two complementary quantum formulations are provided:

* **Binary VQE** — asset *selection* under a cardinality constraint (QUBO → Ising → VQE)
* **Fractional VQE** — long-only *allocation* on the simplex using a constraint-preserving quantum parameterization

All core logic lives in `src/vqe_portfolio/`; notebooks and examples simply call the public API.

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

**Highlights**

* QUBO formulation mapped to an **Ising Hamiltonian**
* Hardware-efficient **RY + CZ ring** ansatz
* VQE minimizes ⟨H⟩ directly
* Outputs include probabilities, samples, Top‑K projections, λ‑sweeps, and efficient frontiers

Notebook client:

* `notebooks/Binary.ipynb`

---

### 2️⃣ Fractional VQE (Continuous Allocation)

Solve the long-only mean–variance problem on the simplex:

$$
\min_{w \in \Delta}
;-\mu^\top w + \lambda, w^\top \Sigma w
\quad\text{with}\quad
\Delta={w\ge0,\sum_i w_i=1}
$$

**Highlights**

* Simplex constraint enforced **by construction**
* No penalty tuning required
* Smooth λ‑sweeps with optional warm starts
* Efficient frontier computed from allocations

Notebook clients:

* `notebooks/Fractional.ipynb`
* `notebooks/examples/Real_Example.ipynb`

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
├── Binary.ipynb
├── Fractional.ipynb
├── examples/
│   └── Real_Example.ipynb
└── images/
```

---

## 📖 Usage

See **[USAGE.md](USAGE.md)** for:

* Minimal API examples
* Synthetic-data quickstart
* Real‑data workflows
* λ‑sweeps and efficient frontiers

---

## 📚 Additional Documentation

* **Theory & derivations**: [`THEORY.md`](THEORY.md)
* **Results & figures**: [`RESULTS.md`](RESULTS.md)

---

## 🧠 Why This Matters

This project demonstrates:

* Mapping **financial optimization problems** to quantum Hamiltonians
* Clean constraint handling (cardinality vs simplex)
* A strict separation between **research code** and **experiment clients**
* Reproducible hybrid quantum–classical workflows
* Production‑grade packaging and CI for quantum algorithms

---

## 🧾 References

* QUBO overview: [https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization)
* PennyLane documentation: [https://docs.pennylane.ai](https://docs.pennylane.ai)

---

**Author**: Sid Richards
GitHub: [@SidRichardsQuantum](https://github.com/SidRichardsQuantum)
MIT License — see [LICENSE](LICENSE)
