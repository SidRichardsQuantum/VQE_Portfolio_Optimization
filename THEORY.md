# Theory

This document describes the mathematical formulation and quantum encodings
used in the `vqe_portfolio` package. It focuses on **how classical portfolio
optimization problems are mapped to quantum Hamiltonians** suitable for
Variational Quantum Eigensolvers (VQE).

---

## Table of Contents

1. [Classical Mean–Variance Portfolio Optimization](#1-classical-meanvariance-portfolio-optimization)
2. [Variational Quantum Eigensolvers (VQE)](#2-variational-quantum-eigensolvers-vqe)
3. [Binary Portfolio Optimization via QUBO](#3-binary-portfolio-optimization-via-qubo)
4. [Fractional Portfolio Optimization via Simplex Encoding](#4-fractional-portfolio-optimization-via-simplex-encoding)
5. [Binary vs Fractional Encodings](#5-binary-vs-fractional-encodings)

---

## 1. Classical Mean–Variance Portfolio Optimization

Let:
- $ \mu \in \mathbb{R}^n $ be the vector of expected returns
- $ \Sigma \in \mathbb{R}^{n \times n} $ be the covariance matrix
- $ w \in \mathbb{R}^n $ be the portfolio weights

The classical mean–variance objective is:

$$
\min_w \; -\mu^\top w + \lambda\, w^\top \Sigma w
$$

subject to constraints such as:
- **Long-only:** $ w_i \ge 0 $
- **Budget:** $ \sum_i w_i = 1 $
- **Cardinality:** $ \|w\|_0 = K $

These constraints motivate different quantum encodings.

---

## 2. Variational Quantum Eigensolvers (VQE)

VQE minimizes the expectation value of a Hamiltonian $ H $ over a
parameterized quantum circuit:

$$
\min_{\theta} \; \langle \psi(\theta) | H | \psi(\theta) \rangle
$$

Key components:
- A parameterized ansatz $ |\psi(\theta)\rangle $
- A classical optimizer updating $ \theta $
- Measurements estimating expectation values

In this project, **the optimization objective itself is encoded into $ H $**.

---

## 3. Binary Portfolio Optimization via QUBO

### 3.1 Problem formulation

Define binary decision variables:
$$
x_i \in \{0,1\}
$$
indicating whether asset $ i $ is selected.

The constrained objective becomes:

$$
\min_{x \in \{0,1\}^n}
\;\lambda\, x^\top \Sigma x
\;-\;\mu^\top x
\;+\;\alpha(\mathbf{1}^\top x - K)^2
$$

where:
- $ \lambda $ controls risk aversion
- $ \alpha $ penalizes violations of the cardinality constraint

This is a **Quadratic Unconstrained Binary Optimization (QUBO)** problem.

---

### 3.2 Ising Hamiltonian mapping

Binary variables are mapped to Pauli-Z operators via:

$$
x_i = \frac{1 - Z_i}{2}
$$

Substituting into the QUBO yields an Ising Hamiltonian of the form:

$$
H = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + \text{const}
$$

This Hamiltonian is implemented in `binary.py` and minimized using VQE.

---

### 3.3 Normalization and sampling

Because expectation values are continuous:
- Raw probabilities are extracted from measurements
- Post-processing (Top-K, mode selection) enforces feasibility

This reflects the fact that **VQE optimizes expectation values, not bitstrings directly**.

---

## 4. Fractional Portfolio Optimization via Simplex Encoding

### 4.1 Continuous parameterization

For fractional allocation, one qubit per asset is used.
Each qubit is prepared with a rotation:

$$
|\psi_i\rangle = RY(\theta_i) |0\rangle
$$

The expectation value of $ Z_i $ defines a nonnegative quantity:

$$
\tilde{w}_i = \frac{1 - \langle Z_i \rangle}{2}
$$

---

### 4.2 Simplex enforcement by construction

Portfolio weights are defined as:

$$
w_i = \frac{\tilde{w}_i}{\sum_j \tilde{w}_j}
$$

This guarantees:
- $ w_i \ge 0 $
- $ \sum_i w_i = 1 $

**No penalty terms are required**.

The VQE objective becomes:

$$
\min_\theta \; -\mu^\top w(\theta) + \lambda\, w(\theta)^\top \Sigma w(\theta)
$$

This encoding is implemented in `fractional.py`.

---

## 5. Binary vs Fractional Encodings

| Aspect | Binary QUBO | Fractional Simplex |
|------|-------------|--------------------|
| Decision type | Discrete selection | Continuous allocation |
| Constraints | Enforced via penalties | Enforced by construction |
| Qubits | $ O(n) $ | $ O(n) $ |
| Output | Probabilistic bitstrings | Deterministic weights |
| Post-processing | Required | Minimal |
| Suitable for | Asset selection | Portfolio allocation |

---

## Summary

This project demonstrates two distinct quantum encodings of portfolio
optimization problems:

- **Binary VQE** emphasizes constraint handling and combinatorial structure
- **Fractional VQE** emphasizes smooth optimization and feasibility by design

Both are implemented within a unified VQE framework, enabling direct
comparison of quantum modeling strategies.

---

## References

The formulations used in this project draw on standard results from
quantum optimization and portfolio theory:

1. Markowitz, H.  
   *Portfolio Selection*, Journal of Finance (1952).

2. Farhi et al.  
   *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028.

3. McClean et al.  
   *The theory of variational hybrid quantum-classical algorithms*,  
   New Journal of Physics 18 (2016).

4. Lucas, A.  
   *Ising formulations of many NP problems*, Frontiers in Physics (2014).

5. PennyLane documentation  
   https://docs.pennylane.ai

---

📘 **Author**: Sid Richards  
MIT License — see [LICENSE](LICENSE)
