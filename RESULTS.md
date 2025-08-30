# Results

## 1. Classical Baseline (Markowitz)

**Diversification Trend (λ sweep)**:

<img src="notebooks/images/Markowitz_Lambda_Sweep.png" alt="Markowitz Lambda Sweep" width="500"/>

## 2. Binary (Selection) Ansatz

Example run ($n=4$, $K=2$, $λ=1.0$):
- Most probable bitstring: `|0101⟩`
- Portfolio decoded: Asset 2 and Asset 4 selected

**Circuit**:

<img src="notebooks/images/Binary_VQE_Circuit.png" alt="Binary Circuit" width="600"/>

**Convergence (iterations vs energy)**:

<img src="notebooks/images/Binary_VQE_Convergence.png" alt="Binary Convergence" width="600"/>

**Sampled Bitstrings (Dirac notation)**:

<img src="notebooks/images/Binary_VQE_Portfolio_Bitstrings.png" alt="Binary Portfolio Bitstrings" width="600"/>

**Marginal Inclusion Probabilities**:

<img src="notebooks/images/Binary_VQE_Probabilities.png" alt="Binary Probabilities" width="600"/>

## 3. Fractional Ansatz

Example run ($n=4$, $λ=0.5$):
  ```
  Asset 1: 0.00%
  Asset 2: 99.92%
  Asset 3: 0.07%
  Asset 4: 0.01%
  ```

**Circuit**:

<img src="notebooks/images/Fractional_VQE_Circuit.png" alt="Fractional Circuit" width="600"/>

**Convergence (iterations vs energy)**:

<img src="notebooks/images/Fractional_VQE_Convergence.png" alt="Fractional Convergence" width="600"/>

**Allocation (Pie)**:

<img src="notebooks/images/Fractional_VQE_Pie.png" alt="Fractional Pie" width="600"/>

**Marginal Probabilities**:

<img src="notebooks/images/Fractional_VQE_Probabilities.png" alt="Fractional Probabilities" width="600"/>

## 4. Real Data Example

Example run (AAPL, MSFT, GOOGL, AMZN; $λ=0.5$):
```
AAPL: 0.00%
MSFT: 0.01%
GOOGL: 0.01%
AMZN: 99.98%
```

**Convergence (iterations vs energy)**:

<img src="notebooks/images/Fractional_Example_Convergence.png" alt="Real Data Convergence" width="600"/>

**Allocation (Pie)**:

<img src="notebooks/images/Fractional_Example_Pie.png" alt="Real Data Pie" width="600"/>

**Marginal Probabilities**:

<img src="notebooks/images/Fractional_Example_Probabilities.png" alt="Real Data Probabilities" width="600"/>

## Comparative Observations

- **Binary QUBO**: enforces discrete picks; slower and noisier convergence.  
- **Fractional VQE**: lighter, faster; suitable for fractional allocation.  
- **Classical baseline**: Markowitz solution closely matches Fractional VQE.

---

📘 Author: Sid Richards ([@SidRichardsQuantum](https://www.linkedin.com/in/sid-richards-21374b30b/))

MIT License – see [LICENSE](LICENSE) for details.
