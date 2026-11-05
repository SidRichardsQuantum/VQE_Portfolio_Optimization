# Small-instance validation

This pilot checks the research pipeline before a larger scaling study. The runner uses library modules, saves trial-level evidence, and generates a Markdown report. Notebooks can consume the saved CSVs without rerunning the algorithms.

## Design

The committed `small_instance_validation.json` specifies:

- 4, 6 and 8 securities, with three independently generated instances at each size (instance seeds 101, 202 and 303).
- Binary cardinality `K = floor(n / 2)` and a separate fractional simplex allocation track.
- Three solver seeds (0, 1 and 2) for each stochastic method on each instance. Deterministic methods run once per instance.
- Binary VQE at depth 2; QAOA with X and XY mixers at depth 1; Fractional VQE as a continuous-allocation control. Each uses 25 optimiser steps.
- Noiseless `default.qubit` execution with exact-expectation training. Binary candidate extraction uses 512 samples, with an explicit Top-K fallback if none are feasible.
- Classical exhaustive binary search, HiGHS integer optimisation, multistart swap search, best of 512 uniformly sampled feasible subsets (`random-k`), numerical active-set fractional optimisation and OSQP. The random baseline checks whether best-of-many candidate extraction alone explains good small-instance results; equal sample counts do not imply equal time budgets.

This produces 18 problem snapshots (nine per track) and 198 solver trials, including 108 quantum-simulator trials. The synthetic factor-model generator samples expected returns, factor exposures and strictly positive idiosyncratic variances. It preserves typical coefficient scales as n increases, but represents only one problem family. Solver seeds are repeated trials, not additional independent instances.

## Run and resume

From the repository root, install the local package and classical extras:

```bash
pip install -e ".[classical]"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_small_instance_validation.py \
  --config experiments/small_instance_validation.json \
  --output-dir results/my_validation
```

For an interrupted experiment, repeat the command with `--resume`. The runner refuses a nonempty output directory without that flag and refuses resume if the configuration, source hashes or recorded environment differ. It saves an atomic checkpoint after every trial. Run only one writer per output directory. Checkpointed errors are not retried silently; use a new output directory after diagnosing them. The exit code is nonzero when any solver exception or reference contradiction is recorded.

Change the configuration and use a new directory for sensitivity experiments. `k_fractions` can contain several fractions (for example, `[0.25, 0.5]`); cardinalities use `max(1, min(n - 1, floor(n * fraction)))` and duplicate cardinalities are collapsed. Fractional instances are included once per dataset, regardless of the number of binary cardinalities. Supported sizes are 2–16; enumeration and simulation become costly toward the upper end.

To run without optional classical solver dependencies, remove `highs` and `osqp` from a copied configuration. Keep the exact references for validation. Higher step counts, circuit depths, shots, risk aversions and penalties should be separate declared sensitivity experiments, rather than tuned on the reported evaluation instances.

## Outputs and interpretation

Each experiment directory contains:

| File | Contents |
| --- | --- |
| `manifest.json` | Resolved settings; data snapshots and hashes; reference portfolios, objectives and preparation runtimes; Python and solver versions; platform and thread settings; source SHA-256 hashes; creation time and Git HEAD |
| `trials.json` | Authoritative resumable checkpoint, including raw candidates, exceptions and validation failures |
| `trials.csv` | Trial-level analysis table, including feasibility, original objective, absolute/normalised gap, runtime, sample feasibility and fallback use |
| `summary.csv` | Results grouped by size, cardinality, objective family and method, averaging instances equally |
| `report.md` | Human-readable descriptive results and limitations |

CSV exports and the report are refreshed when the runner exits normally or receives a Python interruption. After an abrupt process kill, resume from `trials.json` to refresh them.

The independent evaluator recomputes original, penalty-free objectives from raw candidates. Binary and fractional objectives are never ranked against one another. References are computed before evaluating solvers and are not passed to them. Exhaustive search supplies the binary optimum; the fractional reference enumerates supports using floating-point linear algebra and is a numerical reference, not an exact-arithmetic certificate. Reference computation is excluded from algorithm runtime; each timed exact comparator recomputes its solution independently.

The normalised gap is `(f - f_star) / S`, with the positive instance scale S defined in `QUANTUM_ADVANTAGE.md`. Target hits require a feasible portfolio and a gap at most 0.001. Candidates beating a reference by more than `1e-8 * S` are flagged for investigation and are not counted as target hits. Feasibility, target-hit and runtime summaries include all trials. Quality summaries are conditional on feasible output: inspect `quality_instance_count` and failures as well as the mean gap. `sample_feasible_rate` is distinct from returned-portfolio feasibility.

The pilot does not estimate confidence intervals, compare under matched end-to-end time budgets, test noise or measure QPU performance. Its small sample is intended to find correctness and reliability problems, not support statistical advantage claims or extrapolated crossover sizes. The 60-second setting limits only the HiGHS solver; it is not a common deadline. Methods execute serially in a fixed order, so these descriptive runtimes also include order and startup effects. A full study needs the larger instance/run counts, tuning split and statistical analysis prescribed by the research protocol.

See the [saved pilot report](../results/small_instance_validation/report.md) for the first run.

## Before-and-after training diagnostic

To isolate the effect of training, the binary diagnostic applies each full circuit at its initial random parameters and at the final parameters after the pilot's 25 Adam updates. It reuses the pilot's nine frozen binary snapshots, exact references, depths, learning rates, penalty and seeds. There are 81 paired runs: nine problems, three methods and three initialisation seeds. Settings are inherited from the input manifest rather than retuned. Fractional VQE returns an allocation directly and is outside this binary sampling diagnostic.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_training_diagnostic.py \
  --pilot-manifest results/small_instance_validation/manifest.json \
  --output-dir results/my_training_diagnostic
```

The script uses the package's production circuit definitions, initialisation and Adam loop. Tests compare trained parameters and probabilities against the public solvers. Exact probabilities are evaluated on `default.qubit` with no shots, candidate selection or Top-K repair. The frozen optimum and quality threshold are available only to the evaluator, not the optimiser.

For each endpoint it reports:

- `p_feasible`: total probability of an exact-K bitstring.
- `p_target`: total probability of a feasible bitstring meeting the original-objective quality target. This unconditional single-draw probability is the primary measure.
- `p_target_given_feasible`: `p_target / p_feasible`, or blank if feasibility probability is zero. This distinguishes selection quality from constraint satisfaction.
- Expected penalised energy: the objective used in training, including the constraint penalty. This can improve even when `p_target` decreases.

Targets use the same instance scale and normalised gap tolerance (0.001) as the validation pilot. All feasible target bitstrings, including ties and near-optimal portfolios inside that tolerance, contribute to the probability. The evaluator enumerates all bitstrings and checks the frozen reference against the independently computed minimum feasible objective. Uniform-bitstring and uniform-feasible single-draw target probabilities are included as context.

`manifest.json` records inherited settings, frozen problems, environment and source hashes, plus the SHA-256 of the input pilot manifest. `pairs.json` checkpoints raw initial/final parameters, full probability vectors, target indices and metrics after each run. `pairs.csv`, `summary.csv`, `report.md` and `training_probability.png` provide analysis outputs. Bitstring vector order is lexicographic, with wire 0 the most significant bit. The inherited `shots_sample` setting is recorded for traceability but is not executed.

Use `--resume` with the same input, sources and environment to complete missing pairs. Failed pairs remain recorded and contribute to error counts; probability means include successful pairs only, with coverage recorded in `quality_instance_count`. Only one process should write to an output directory. Completed pilot artifacts are preserved; the diagnostic writes to its own directory.

The primary comparison is the paired change in `p_target`, averaged over seeds within each instance and then equally across instances. Improved/worsened counts use a `1e-12` numerical deadband, not a significance test. These exact probability evaluations have no shot noise, but conclusions still depend on the small instance and initialisation sample. There are no shot/depth/training-budget sweeps, confidence intervals, QPU measurements or advantage claims in this diagnostic.

See the [saved diagnostic report](../results/training_diagnostic/report.md).
