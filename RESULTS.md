# Results

## 1. Binary (Selection) Ansatz

Example run ($n=4$, $K=2$, $λ=1.0$):
- Most probable bitstring: `|0101⟩`
- Portfolio decoded: Asset 2 and Asset 4 selected

**Circuit**
![Binary Circuit](images/Binary_VQE_Circuit.png)

**Convergence (iterations vs cost)**
![Binary Convergence](images/Binary_VQE_Convergence.png)

**Sampled Bitstrings (Dirac notation)**
![Binary Portfolio Bitstrings](images/Binary_VQE_Portfolio_Bitstrings.png)

**Marginal Inclusion Probabilities**
![Binary Probabilities](images/Binary_VQE_Probabilities.png)

## 2. Fractional Ansatz

Example run ($n=4$, $λ=0.5$). Final weights:
  ```
  Asset 1: 0.00%
  Asset 2: 99.92%
  Asset 3: 0.07%
  Asset 4: 0.01%
  ```

**Circuit**
![Fractional Circuit](images/Fractional_VQE_Circuit.png)

**Convergence**
![Fractional Convergence](images/Fractional_VQE_Convergence.png)

**Allocation (Pie)**
![Fractional Pie](images/Fractional_VQE_Pie.png)

**Marginal Probabilities**
![Fractional Probabilities](images/Fractional_VQE_Probabilities.png)

## 3. Real Data Example

Tickers: AAPL, MSFT, GOOGL, AMZN (2023–2024 via Yahoo Finance).  
Fractional VQE run ($λ=0.5$) produced:
```
AAPL: 0.00%
MSFT: 0.01%
GOOGL: 0.01%
AMZN: 99.98%
```

**Convergence**
![Real Data Convergence](images/Fractional_Example_Convergence.png)

**Allocation (Pie)**
![Real Data Pie](images/Fractional_Example_Pie.png)

**Marginal Probabilities**
![Real Data Probabilities](images/Fractional_Example_Probabilities.png)

## 4. Comparative Observations

- **Binary QUBO**: enforces discrete picks; slower and noisier convergence.  
- **Fractional VQE**: lighter, faster; suitable for fractional allocation.  
- **Classical baseline**: Markowitz solution closely matches Fractional VQE.

---

📘 Author: Sid Richards ([@SidRichardsQuantum](https://www.linkedin.com/in/sid-richards-21374b30b/))

MIT License – see [LICENSE](LICENSE) for details.
