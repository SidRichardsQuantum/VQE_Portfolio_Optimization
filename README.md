# Portfolio Optimization via VQE

<p align="center">

<a href="https://pypi.org/project/vqe-portfolio/">
<img src="https://img.shields.io/pypi/v/vqe-portfolio?style=flat-square" alt="PyPI Version">
</a>

<a href="https://pypi.org/project/vqe-portfolio/">
<img src="https://img.shields.io/pypi/pyversions/vqe-portfolio?style=flat-square" alt="Python Versions">
</a>

<a href="https://github.com/SidRichardsQuantum/VQE_Portfolio_Optimization/actions/workflows/ci.yml">
<img src="https://img.shields.io/github/actions/workflow/status/SidRichardsQuantum/VQE_Portfolio_Optimization/ci.yml?label=tests&style=flat-square" alt="Tests">
</a>

<a href="https://github.com/SidRichardsQuantum/VQE_Portfolio_Optimization/blob/main/LICENSE">
<img src="https://img.shields.io/github/license/SidRichardsQuantum/VQE_Portfolio_Optimization?style=flat-square" alt="License">
</a>

<a href="https://github.com/sponsors/SidRichardsQuantum">
<img src="https://img.shields.io/badge/sponsor-GitHub-ea4aaa?style=flat-square&logo=githubsponsors" alt="Sponsor">
</a>

</p>

PyPI: [https://pypi.org/project/vqe-portfolio/](https://pypi.org/project/vqe-portfolio/)

Website: [https://sidrichardsquantum.github.io/VQE_Portfolio_Optimization/](https://sidrichardsquantum.github.io/VQE_Portfolio_Optimization/)

A Python library for portfolio optimization with PennyLane. It provides Binary VQE and QAOA for selecting a fixed number of assets, and a variational quantum allocator for long-only allocation. Classical baselines and independent candidate evaluation support reproducible comparisons.

The allocator was historically called "Fractional VQE". It parameterises weights
with a quantum circuit and evaluates the portfolio cost classically; it is not
Hamiltonian-expectation VQE. Use `run_variational_allocator` and
`VariationalAllocatorConfig`; the old Python and CLI names remain aliases.

The experiments use quantum simulation and do not establish quantum advantage or production readiness.

## Installation

```bash
pip install vqe-portfolio
```

Requires Python 3.10+ and PennyLane 0.42.3+. CI tests the minimum core
dependencies and current compatible dependencies on Python 3.10–3.12.

Optional dependencies:

```bash
pip install "vqe-portfolio[data]"       # Market data utilities
pip install "vqe-portfolio[classical]"  # HiGHS, OSQP, and SCIP baselines
pip install "vqe-portfolio[scip]"       # SCIP only
pip install -e ".[dev,docs]"           # Development and documentation
```

## Quickstart

Run a small synthetic asset-selection problem:

```python
import numpy as np
from vqe_portfolio import BinaryVQEConfig, run_binary_vqe

mu = np.array([0.10, 0.12, 0.07, 0.09])
sigma = np.eye(4) * 0.02
config = BinaryVQEConfig(k=2, lam=4.0, alpha=2.0, steps=25, seed=0)
result = run_binary_vqe(mu, sigma, config)
print(result.x_topk)
```

The command-line interface supports the same solvers:

```bash
vqe-portfolio --help
python -m vqe_portfolio --help
```

See the [usage guide](USAGE.md) for complete API and CLI examples, ansatz options, real-data workflows, and efficient frontiers.

Classical exact-K comparators include NumPy-only multistart swap search,
simulated annealing, tabu search, memetic genetic search, iterated local search
(ILS), GRASP, and variable neighbourhood search (VNS), plus optional QP relaxation with rounding and swap refinement,
HiGHS linearised integer and SCIP quadratic integer optimisation. All support end-to-end execution
budgets; see [classical baselines](USAGE.md#strong-classical-baselines).

NumPy projected gradient descent supplies an iterative classical comparator for
the separate long-only fractional allocation track, alongside optional OSQP.

## Repository structure

```text
src/vqe_portfolio/
├── binary.py, qaoa.py, fractional.py   # Quantum solvers
├── problem.py, types.py               # Problem evaluation and solver types
├── ansatz.py, optimize.py             # Shared circuit and optimizer code
├── baselines/                        # Classical references and optional solvers
├── benchmark.py                      # Independent candidate evaluation/reporting
├── research/                         # Study logic and shared runner infrastructure
└── ...                               # CLI, data, plotting, frontier utilities
scripts/                              # Experiment entry points and report generation
experiments/                          # Study configuration and reproduction guide
results/                              # Saved research evidence
notebooks/                            # Tutorial and comparison clients
tests/                               # Solver, research, CLI, compatibility tests
docs/                                 # Sphinx documentation
```

See [architecture and development](docs/architecture.md) for module boundaries and the v0.3.0 migration notes. Existing public imports remain supported. Version 0.3.1 adds opt-in solver reliability controls and end-to-end execution budgets; see the [usage guide](USAGE.md#solver-reliability-controls-v031) and [changelog](CHANGELOG.md).

Version 0.3.3 adds simulated annealing, tabu search, and SCIP to the classical
comparisons, small-instance validation, and budget CLI. Version 0.3.4 adds
memetic genetic search, ILS, GRASP, QP rounding, VNS, and projected gradient
descent; fixes selection and allocator boundaries; and introduces the canonical
allocator names and individual algorithm guides. Archived research results
retain their original methods and provenance.

## Documentation and research

| Guide | Contents |
| --- | --- |
| [Usage](USAGE.md) | Installation, API, CLI, benchmarks, and reproduction commands |
| [Theory](THEORY.md) | Portfolio formulations, circuit mappings, and derivations |
| [Algorithms](docs/algorithms/index.md) | Concise descriptions of quantum and classical methods |
| [Results](RESULTS.md) | Committed benchmark and diagnostic findings |
| [Research protocol](QUANTUM_ADVANTAGE.md) | Evaluation design and limits on advantage claims |
| [Experiments](experiments/README.md) | Validation, training, sampling, and failure diagnostics |

Start with `notebooks/Binary.ipynb`, `notebooks/QAOA.ipynb`, or `notebooks/Fractional.ipynb`. Real-data examples are in `notebooks/examples/`.

Saved study reports:

- [Small-instance validation](results/small_instance_validation/report.md)
- [Allocator correction validation](results/allocator_validation/report.md)
- [Before/after training](results/training_diagnostic/report.md)
- [Sampling allowances](results/sampling_allowance/report.md)
- [Training effort](results/training_effort/report.md)
- [Failure diagnostics](results/failure_diagnostics/report.md)
- [Solver reliability controls](results/solver_reliability/report.md)
- [Fixed-deadline simulator pilot](results/budget_benchmark/report.md)
- [Fresh-instance deadline comparison](results/fresh_instance_budget/report.md)

## Author and support

[Sid Richards](https://github.com/SidRichardsQuantum) · [LinkedIn](https://www.linkedin.com/in/sid-richards-21374b30b/) · [Sponsor development](https://github.com/sponsors/SidRichardsQuantum)

MIT License — see [LICENSE](https://github.com/SidRichardsQuantum/VQE_Portfolio_Optimization/blob/main/LICENSE).
