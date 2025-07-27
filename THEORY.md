# Theory: VQE for Portfolio Optimization

## 1. Classical Portfolio Optimization

- Expected return, risk, and the mean-variance framework.
- Typical constraints (e.g., sum of weights = 1, budget, cardinality).

## 2. Quantum Reformulation via VQE

- Overview of VQE: variational circuits + classical optimizer.
- Objective: Minimize the expectation value of a Hamiltonian.

## 3. Binary QUBO Encoding

- Bitstring representation using multiple qubits per asset.
- Ising Hamiltonian mapping (return, risk, and penalty terms).
- Why normalization is needed.

## 4. Fractional Ansatz Encoding

- Continuous-variable representation using RY(θ) gates.
- Interpreting sin²(θ/2) as investment weights.
- Advantage: fewer qubits, no sampling.

## 5. Comparison: Binary vs. Fractional

| Aspect         | Binary QUBO | Fractional Ansatz |
|----------------|-------------|-------------------|
| Qubits needed  | High        | Low               |
| Output type    | Discrete    | Continuous        |
| Encoding       | Bitstring   | Angle rotation    |
| Performance    | Slower      | Faster            |
