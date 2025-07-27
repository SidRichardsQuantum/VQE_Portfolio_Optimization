# VQE_Portfolio_Optimization

Optimizing asset portfolios using a Variational Quantum Eigensolver (VQE) in a hybrid quantum-classical algorithm.

This repo implements two methods:

- `VQE_Portfolio_BinaryEncoding.ipynb` – a QUBO-style binary encoding using multi-qubit representations.
- `VQE_Portfolio_Fractional.ipynb` – a fractional ansatz using parameterized RY rotations.

---

## 📚 Documentation

- **Theory**: Full derivation and model design in [`THEORY.md`](THEORY.md)
- **Results**: Output portfolios, convergence plots and insights in [`RESULTS.md`](RESULTS.md)

---

## 🧠 Overview

Portfolio optimization is framed as a QUBO problem, then mapped to a quantum Hamiltonian via PennyLane’s tools. The VQE algorithm minimizes the expectation value of the Hamiltonian to find optimal allocations.

---

## 🧾 References

- [QUBO – Wikipedia](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization)
- PennyLane docs: https://docs.pennylane.ai

---

📘 Author: Sid Richards ([@SidRichardsQuantum](https://www.linkedin.com/in/sid-richards-21374b30b/))

MIT License – see [LICENSE](LICENSE) for details.
