# Theory

This document describes the mathematical formulation and quantum encodings used in the `vqe_portfolio` package.
It focuses on **how classical portfolio optimization problems are mapped to quantum Hamiltonians** suitable for hybrid quantum–classical algorithms such as **VQE** and **QAOA**.

---

## Table of Contents

- [Classical Mean–Variance Portfolio Optimization](#1-classical-meanvariance-portfolio-optimization)

- [Variational Quantum Eigensolvers (VQE)](#2-variational-quantum-eigensolvers-vqe)

- [Quantum Approximate Optimization Algorithm (QAOA)](#3-quantum-approximate-optimization-algorithm-qaoa)

  - [Cost Hamiltonian](#31-cost-hamiltonian)
  - [Mixer Hamiltonians](#32-mixer-hamiltonians)
  - [Sampling interpretation](#33-sampling-interpretation)

- [Binary Portfolio Optimization via QUBO](#4-binary-portfolio-optimization-via-qubo)

  - [Binary decision variables](#41-binary-decision-variables)
  - [Ising mapping](#42-ising-mapping)
  - [Expectation vs bitstrings](#43-expectation-vs-bitstrings)

- [Fractional Portfolio Optimization via Simplex Encoding](#5-fractional-portfolio-optimization-via-simplex-encoding)

  - [Continuous parameterization](#51-continuous-parameterization)
  - [Simplex normalization](#52-simplex-normalization)
  - [Optimization objective](#53-optimization-objective)

- [Binary vs Fractional Encodings](#6-binary-vs-fractional-encodings)

- [Summary](#summary)

- [References](#references)

- [Author](#author)

- [License](#license)

---

## 1. Classical Mean–Variance Portfolio Optimization

Let:

- $ \mu \in \mathbb{R}^n $ be expected returns
- $ \Sigma \in \mathbb{R}^{n \times n} $ be the covariance matrix
- $ w \in \mathbb{R}^n $ be portfolio weights

The classical mean–variance objective is:

$$
\min_w ; -\mu^\top w + \lambda, w^\top \Sigma w
$$

where:

- $ \lambda > 0 $ controls risk aversion

Typical constraints include:

#### Long-only constraint

$$
w_i \ge 0
$$

#### Budget constraint

$$
\sum_i w_i = 1
$$

#### Cardinality constraint

$$
|w|_0 = K
$$

Cardinality constraints introduce **combinatorial structure**, motivating discrete encodings.

---

## 2. Variational Quantum Eigensolvers (VQE)

VQE solves optimization problems of the form:

$$
\min_\theta ;
\langle \psi(\theta) | H | \psi(\theta) \rangle
$$

where:

- $ H $ is a problem Hamiltonian
- $ |\psi(\theta)\rangle $ is a parameterized quantum state
- $ \theta $ are circuit parameters optimized classically

Algorithm structure:

1. prepare ansatz state
2. measure expectation value
3. update parameters using classical optimizer
4. repeat until convergence

VQE is particularly useful when:

- objective functions can be expressed as expectation values
- constraints can be encoded into Hamiltonians
- gradients can be estimated efficiently

In this project, the portfolio objective is encoded directly into $ H $.

---

## 3. Quantum Approximate Optimization Algorithm (QAOA)

QAOA is a gate-based algorithm designed for combinatorial optimization problems.

It alternates between:

- cost evolution
- mixing evolution

The QAOA state is:

$$
|\psi(\boldsymbol{\gamma},\boldsymbol{\beta})\rangle
=
\prod_{\ell=1}^p
e^{-i\beta_\ell H_M}
e^{-i\gamma_\ell H_C}
|+\rangle^{\otimes n}
$$

where:

- $ H_C $ is the cost Hamiltonian
- $ H_M $ is the mixer Hamiltonian
- $ p $ is the circuit depth

Parameters:

- $ \boldsymbol{\gamma} = (\gamma_1,\dots,\gamma_p) $
- $ \boldsymbol{\beta} = (\beta_1,\dots,\beta_p) $

are optimized using classical optimization.

---

### 3.1 Cost Hamiltonian

Portfolio optimization produces a QUBO objective:

$$
C(x)
=
\lambda x^\top \Sigma x - \mu^\top x + \alpha(\mathbf{1}^\top x - K)^2
$$

which maps to an Ising Hamiltonian:

$$
H_C = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + \text{const}
$$

This Hamiltonian is shared with Binary VQE.

---

### 3.2 Mixer Hamiltonians

Two mixers are implemented.

#### X mixer

$$
H_M^{(X)}
=
\sum_i X_i
$$

Promotes exploration across the full computational basis.

#### XY mixer

$$
H_M^{(XY)}
=
\sum_{i<j}
(X_i X_j + Y_i Y_j)
$$

Preserves approximate Hamming-weight structure, making it useful for:

- cardinality-constrained problems
- constrained combinatorial search spaces

---

### 3.3 Sampling interpretation

QAOA produces a probability distribution over bitstrings:

$$
p(x)
=
|\langle x|\psi\rangle|^2
$$

From this distribution we compute:

#### Marginal probabilities

$$
p_i = \mathbb{E}[x_i]
$$

#### Top-K projection

Select the $ K $ largest marginals.

#### Mode bitstring

Most frequently sampled bitstring.

#### Best feasible candidate

Lowest-cost bitstring satisfying the constraint.

QAOA therefore provides:

- probabilistic solutions
- candidate discrete portfolios
- insight into landscape structure

---

## 4. Binary Portfolio Optimization via QUBO

### 4.1 Binary decision variables

Let:

$$
x_i \in {0,1}
$$

indicate asset inclusion.

Objective:

$$
\min_{x \in {0,1}^n}
\lambda x^\top \Sigma x
-
\mu^\top x
+
\alpha(\mathbf{1}^\top x - K)^2
$$

This is a Quadratic Unconstrained Binary Optimization (QUBO) problem.

---

### 4.2 Ising mapping

Binary variables are mapped to Pauli-Z operators:

$$
x_i = \frac{1 - Z_i}{2}
$$

Substitution yields:

$$
H
=

\sum_i h_i Z_i
+
\sum_{i<j} J_{ij} Z_i Z_j
+
\text{const}
$$

This Hamiltonian is minimized via:

- VQE
- QAOA

---

### 4.3 Expectation vs bitstrings

Variational algorithms optimize expectation values:

$$
\min_\theta
\langle H \rangle
$$

but practical portfolios require discrete solutions.

Post-processing extracts:

- deterministic selections
- feasible bitstrings
- empirical distributions

---

## 5. Fractional Portfolio Optimization via Simplex Encoding

### 5.1 Continuous parameterization

Each qubit prepares:

$$
|\psi_i\rangle = RY(\theta_i)|0\rangle
$$

Measurement produces:

$$
\tilde{w}_i
=
\frac{1 - \langle Z_i \rangle}{2}
$$

---

### 5.2 Simplex normalization

Weights:

$$
w_i
=
\frac{\tilde{w}_i}{\sum_j \tilde{w}_j}
$$

Properties:

- $ w_i \ge 0 $
- $ \sum_i w_i = 1 $

Constraints are satisfied by construction.

---

### 5.3 Optimization objective

$$
\min_\theta
-
\mu^\top w(\theta)
+
\lambda w(\theta)^\top \Sigma w(\theta)
$$

Advantages:

- smooth landscape
- no penalty tuning
- deterministic feasible solutions

---

## 6. Binary vs Fractional Encodings

| aspect              | binary (VQE / QAOA) | fractional VQE |
| ------------------- | ------------------- | -------------- |
| decision space      | discrete            | continuous     |
| constraint handling | penalty             | structural     |
| objective landscape | non-convex          | smooth         |
| output              | bitstrings          | weights        |
| sampling required   | yes                 | no             |
| post-processing     | required            | minimal        |
| suitable for        | asset selection     | allocation     |

---

## Summary

This project demonstrates three complementary quantum approaches:

#### Binary VQE

- Hamiltonian expectation minimization
- flexible ansatz design
- probabilistic bitstring outputs

#### QAOA

- structured alternating operators
- natural fit for QUBO problems
- interpretable circuit depth parameter

#### Fractional VQE

- continuous parameterization
- exact feasibility
- efficient frontier construction

Together they provide a consistent framework for studying how portfolio optimization behaves under quantum-native representations.

---

## References

1. Markowitz, H.
   *Portfolio Selection*, Journal of Finance (1952)

2. Farhi et al.
   *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028

3. McClean et al.
   *The theory of variational hybrid quantum-classical algorithms*,
   New Journal of Physics 18 (2016)

4. Lucas, A.
   *Ising formulations of many NP problems*, Frontiers in Physics (2014)

5. PennyLane documentation
   [https://docs.pennylane.ai](https://docs.pennylane.ai)

---

## Author

Sid Richards

LinkedIn
[https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

GitHub
[https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

---

## License

MIT License — see [LICENSE](LICENSE)
