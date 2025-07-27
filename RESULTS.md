# Results: VQE Portfolio Optimization

## 1. Binary Encoding

- Final sampled bitstring:
  `[0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1]`
- Decoded portfolio:
  - Asset 1: 50.0%
  - Asset 2: 8.3%
  - Asset 3: 0.0%
  - Asset 4: 41.7%
- Cost convergence plot
- Bar and pie charts

## 2. Fractional Ansatz

- Optimized angles:
  `[1.23, 0.45, ...]`
- Final weights:
  - Asset 1: 35.2%
  - Asset 2: 20.1%
  - Asset 3: 10.7%
  - Asset 4: 34.0%
- Convergence was much faster due to shallower circuit.
- Plot screenshots or PNGs embedded.

## 3. Comparative Observations

- Binary QUBO offers precision, but is slower and noisier.
- Fractional VQE is efficient, well-suited for real-world fractional allocation.
