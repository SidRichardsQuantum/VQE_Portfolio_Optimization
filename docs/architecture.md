# Architecture and development

The package separates portfolio solvers, independent evaluation, and research
workflows. Notebooks call the library, and scripts expose experiment arguments
and generate study reports.

## Module boundaries

| Location | Responsibility |
| --- | --- |
| `problem.py` | Problem snapshots, original objectives, and independent feasibility evaluation |
| `types.py` | Quantum solver configurations and results |
| `binary.py`, `qaoa.py`, `fractional.py` | Quantum solver implementations |
| `ansatz.py`, `optimize.py`, `reliability.py` | Shared circuits, optimizer loop, and reference-free reliability controls |
| `execution.py` | Isolated-process deadlines, incumbent delivery, and common termination metadata |
| `baselines/reference.py` | NumPy reference enumeration and simple heuristics |
| `baselines/solvers.py` | NumPy search heuristics, optional classical solvers, and termination metadata |
| `benchmark.py` | Candidate trial records, independent evaluation, and common reporting helpers |
| `research/comparison.py` | Synthetic and market-data comparison orchestration |
| `research/validation.py` | Small-instance preparation, trial planning, and validation |
| `research/training_diagnostics.py` | Initial/trained circuit distribution analysis |
| `research/training_effort.py` | Continuous training trajectories and paired comparisons |
| `research/sampling_allowance.py` | Analytical sampling success from frozen distributions |
| `research/failure_diagnostics.py` | Reachability, depth, and penalty analysis |
| `research/reliability.py`, `research/budget.py` | Controlled solver comparisons and fixed-deadline trial evaluation |
| `research/infrastructure.py` | Provenance, atomic JSON checkpoints, and CSV exports |
| `scripts/` | CLI entry points, study-specific plots, and report generation |

Baseline implementations depend on problem definitions, rather than experiment
orchestration. Research workflows compose solvers and evaluation helpers. Shared
comparison functions belong in the library so scripts do not depend on each
other. Optional market-data and classical solver dependencies load when used.

## Imports and v0.3.0 migration

The descriptive allocation API is `run_variational_allocator`,
`allocator_lambda_sweep`, `VariationalAllocatorConfig`, and
`VariationalAllocatorResult`. Historical fractional-VQE imports are exact
aliases. CLI aliases and old experiment method IDs remain accepted; newly
generated labels identify the variational quantum allocator. Its cost is a
classical function of circuit expectations, not Hamiltonian-expectation VQE.
The corrected smoothed normalisation requires fresh experiment outputs;
historical artifacts keep their original labels and source hashes.

The package-root API remains supported:

```python
from vqe_portfolio import BinaryVQEConfig, run_binary_vqe
from vqe_portfolio.baselines import exhaustive_binary_baseline
from vqe_portfolio.benchmark import CandidateTrial, comparison_rows
from vqe_portfolio.research.validation import ValidationConfig
```

Version 0.3.0 reorganizes implementation modules without changing solver
signatures, CLI arguments, or saved study schemas:

- The former top-level validation and diagnostic modules are compatibility
  aliases to the corresponding `research` modules.
- `vqe_portfolio.classical` aliases `vqe_portfolio.baselines.solvers`.
- `vqe_portfolio.comparison` re-exports its original baseline, objective,
  formatting, and real-data comparison functions. Prefer the locations above
  for new code.
- Shared comparison-building functions previously imported from
  `scripts.generate_comparison_results` now live in `research.comparison`.
  The script retains those helper exports for existing clients.

New manifests include recursive package source hashes and explicitly identify
the active runner. Existing committed evidence retains its historical hashes.
Resume still requires matching code, settings, and environment; use a new
output directory after upgrading, or the original revision to resume old runs.
Frozen version-1 study inputs remain readable by downstream analyses.

## Development checks

From the repository root, install the package and relevant extras:

```bash
pip install -e ".[dev,docs,classical,data,notebooks]"
ruff check .
black --check src tests scripts
pytest -q
python -m sphinx -W -b html docs docs/_build/html
pytest --nbmake notebooks/Binary.ipynb
python -m build
python -m twine check dist/*
```

Base-install tests should also pass without optional classical and data
dependencies. CI checks Python 3.10, 3.11, and 3.12, builds the documentation,
executes a notebook smoke test, and validates the wheel and source distribution.
It also runs the suite with NumPy 1.26.0, Matplotlib 3.8.0, and PennyLane 0.42.3
on each supported Python version. PennyLane 0.42.3 is the minimum supported
release for the solver shot-setting API and its dependency compatibility fixes.
PyPI publishing reuses the complete CI workflow before building artifacts and
requires a tag matching the package version, including manually triggered runs.

## Documentation sources

The root `README.md`, `USAGE.md`, `THEORY.md`, `RESULTS.md`, and
`QUANTUM_ADVANTAGE.md` are the guide sources. Sphinx includes each guide once;
its navigation uses `index`, `USAGE`, `THEORY`, `results`, and
`QUANTUM_ADVANTAGE`. This page documents package organization and contributor
checks directly in Sphinx.

## v0.3.1 execution and reliability

Solver defaults retain the legacy circuit and penalty choices. Explicit XY
initial-state/topology options and Binary VQE penalty, restart, normalization,
best-iterate, and learning-rate controls support controlled comparisons.
`run_budgeted` owns the end-to-end wall clock and enforces deadlines in spawned
processes; only independently valid candidates received on time can become
incumbents. Reference targets are used by research evaluation after execution,
never by optimizers or worker dispatch. See the [usage guide](USAGE.md) for timing
boundaries, callback overhead, and spawn entry-point requirements.

Optimizer logs now evaluate the post-update cost matching each snapshot and
always include the last update. Historical evidence is preserved; changed costs
and finite-shot random streams require fresh study output directories.

Budget results also record worker phase durations and overlapping delivery and
parent-validation diagnostics. Solvers accept an optional `phase_callback` for
this instrumentation; the ordinary optimizer and candidate APIs retain their
defaults. Untrained XY dispatch executes production sampling circuits repeatedly
at the original random parameters. QAOA permits zero training updates for initial
state inspection. The fresh-instance runner resolves named presets, rejects
development-instance overlap, and keeps variants separate in summaries and
descriptive pairs. Historical manifests and reports remain unchanged.
