# Small-instance validation

Current allocator terminology is **variational quantum allocator**. The Python
API is `run_variational_allocator` with `VariationalAllocatorConfig`; historical
fractional-VQE names remain aliases. New method IDs use `variational-allocator`,
while copied historical configurations with `fractional-vqe` remain accepted.
New default method sets include the allocator once. Historical reports below
retain the old labels and original parameterisation.

To repeat the correctness comparison after the normalisation and naming fixes:

```bash
python scripts/run_small_instance_validation.py \
  --config experiments/allocator_validation.json \
  --output-dir results/allocator_validation
```

This uses 4 and 6 assets, two independent instance seeds and two solver seeds,
15 quantum optimisation steps, and 256 binary extraction samples. It includes
the corrected top-return control, the current quantum methods, and all added
classical comparators. Binary and fractional references remain separate. Unit
tests cover K=0 and zero/near-zero measured mass; this study exercises normal
instances with the corrected map. It is a descriptive validation, with no
matched-time, scaling, statistical-significance, or quantum-advantage claim.

The saved [allocator validation report](../results/allocator_validation/report.md)
completed all 128 trials with feasible returned portfolios, no solver errors,
and no reference contradictions. Numerical quality is reported separately;
the allocator hit its fractional target in 2 of 8 trials. The saved manifest,
JSON checkpoint, and CSV exports retain frozen inputs and source/environment
provenance. Earlier study artifacts remain unchanged.

Study implementations live in `vqe_portfolio.research`. The executable scripts use shared provenance, atomic checkpoint writing, and CSV exports from `vqe_portfolio.research.infrastructure`; no runner imports another runner. Classical reference and optional solver implementations live in `vqe_portfolio.baselines`, and independent candidate reporting lives in `vqe_portfolio.benchmark`.

As of v0.3.0, new manifests hash package sources recursively, including subpackages, and record the active runner and installed package version. Saved results retain their original manifests and source hashes. Pre-refactor checkpoints cannot be resumed with changed source code: use the original revision/environment to resume them, or a new output directory for v0.3.0. Downstream studies can still read the frozen schema-version-1 evidence.

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

Version 0.3.3 also supports `anneal`, `tabu`, and `scip` in copied validation configurations and the budget runner's `--methods` selection. Annealing and tabu search run once per solver seed; SCIP runs once per validation instance and records its native status, numerical bound, and gap. New manifests include the PySCIPOpt package version. Use a fresh output directory after changing methods or upgrading; the archived pilot does not include these additions.

Version 0.3.4 adds `memetic`, `ils`, and `grasp` to both runners. Each is a seeded NumPy-only heuristic and runs once per solver seed in validation. `local_restarts` counts independent memetic populations, ILS starts, or GRASP constructions. Algorithm-specific defaults are documented in [usage](../USAGE.md#strong-classical-baselines); the budget runner uses the public helpers' defaults, including generation, perturbation, and construction settings. These methods preserve the existing exact-K objective and are absent from the archived pilot.

`qp-round` is a further deterministic binary heuristic: solve the box relaxation with sum(z)=K using CVXPY/OSQP, round to K assets, then refine with swaps. It runs once per validation instance and reports no certified bound or MIP gap. The helper message, comparison CSV notes, and validation `solver_message` retain numerical relaxation diagnostics. It also supports budgeted execution, publishing improving binary incumbents after the relaxation returns. Install `.[classical]` or `.[markowitz]`; no new dependencies are introduced. It is absent from the archived experiments.

NumPy-only `vns` and `pgd` are also available. VNS is an exact-K binary heuristic run per solver seed, with `local_restarts` controlling its starts. PGD is a deterministic fractional simplex solver run once per validation instance; its first-order stopping diagnostic is saved in `solver_message`. Both publish improving feasible incumbents in budgeted runs, use their documented public defaults, and report no certified bound or MIP gap. Keep their objective tracks separate and use a fresh output directory. Neither is included in the archived pilot.

To run without optional classical solver dependencies, remove `highs`, `osqp`, and (if added) `scip` and `qp-round` from a copied configuration. Keep the exact references for validation. Higher step counts, circuit depths, shots, risk aversions and penalties should be separate declared sensitivity experiments, rather than tuned on the reported evaluation instances.

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

## Sampling allowance diagnostic

This follow-up reuses `manifest.json` and `pairs.json` from the completed training diagnostic. It calculates the probability of finding at least one feasible target-quality portfolio in 16, 64 or 512 independent draws, for the initial circuit, trained circuit and uniform feasible classical sampling. It performs no training and draws no random samples.

```bash
python scripts/run_sampling_allowance.py \
  --diagnostic-dir results/training_diagnostic \
  --draws 16,64,512 \
  --output-dir results/my_sampling_allowance
```

For a fixed single-draw target probability p, the success probability is `1 - (1 - p)^m`. The implementation evaluates the numerically stable equivalent `-expm1(m * log1p(-p))`, handling p=0 and p=1 explicitly. It transforms each run's probability separately, computes the paired training change, averages seeds within each instance, and finally averages instances equally. Transforming a mean p instead would overstate success for heterogeneous runs.

Every raw circuit sample consumes one draw, including infeasible bitstrings. The analysis does not condition on feasibility or apply Top-K repair. Classical sampling draws uniformly from exact-K subsets with replacement, matching the earlier `random-k` baseline's sampling distribution. Its target probability is independently computed from the frozen target subset count and feasible subset count. A classical baseline is recorded once per instance and allowance, rather than repeated across quantum methods and seeds.

The input checkpoint must contain every planned pair exactly once. Snapshots, references, pair metadata, target masks, distribution normalisation and stored target probabilities are checked before producing results. Failed source pairs remain represented with blank probabilities and error counts; conditional coverage is explicit. Source input files are preserved and both are hashed in the new manifest.

The default analysis produces 243 per-pair allowance rows, 27 method/size/allowance summary rows, 27 classical instance/allowance rows and nine classical summaries. Outputs are:

| File | Contents |
| --- | --- |
| `manifest.json` | Draw counts, target tolerance, input-file hashes, software and analysis source hashes; records that no training or sampling occurred |
| `pairs.csv` | Initial/trained success probabilities and paired changes for each saved run and allowance |
| `summary.csv` | Instance-weighted method means, error counts and valid-pair coverage |
| `baseline_trials.csv` | One uniform feasible classical row per instance/allowance |
| `baseline_summary.csv` | Classical means across all instances at each size/allowance |
| `report.md`, `sampling_success.png` | Descriptive tables and success-probability curves |

Use a new output directory when repeating the analysis; it finishes quickly and needs no resume mechanism. The plotted points are the evaluated allowances; lines between them guide the eye. These are analytical expected success probabilities, not newly observed hit rates. Near-100% values may round to 100% in the report; CSVs retain full precision.

The model assumes independent draws from a fixed distribution with replacement. It excludes training and circuit execution costs, hardware noise, drift and postprocessing. Equal draw counts are not equal time or cost budgets, so this study cannot establish quantum advantage. Baseline curves cover all source instances; if source errors occur, quantum curves condition on successful pairs and can have different coverage. The matched classical column in `summary.csv` uses the same coverage as each method.

See the [saved sampling allowance report](../results/sampling_allowance/report.md).

## Training-effort development study

This study compares checkpoints at 0, 25 and 100 optimiser steps on the same nine frozen binary problems, three methods and three initialisation seeds: 81 trajectories and 243 checkpoint distributions. Each trajectory initialises once, runs Adam continuously to step 100, and copies parameters at the requested checkpoints without resetting Adam's moment estimates. Circuit depths, penalties, learning rates, target tolerance and initialisation are inherited from the earlier diagnostic. These are development instances, not a held-out tuning evaluation.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_training_effort.py \
  --diagnostic-dir results/training_diagnostic \
  --steps 0,25,100 --draws 16,64,512 \
  --output-dir results/my_training_effort
```

The runner validates the source checkpoint and requires the new step-zero and step-25 parameters/probabilities to reproduce the saved diagnostic. Target references are available only to evaluation. It records exact target and feasibility probability, target probability conditional on feasibility, expected penalised energy, analytical success with each sampling allowance, and cumulative training time. Success probabilities are calculated per run before instance-weighted aggregation. All checkpoint intervals receive paired comparisons, including improved/worsened run counts with a `1e-12` numerical probability deadband.

Training time is measured only inside the continuous Adam loop. It includes callback and logging overhead, but excludes encoding, device setup and diagnostic probability evaluation. All distributions are evaluated after optimisation finishes, so diagnostic state measurements do not inflate later checkpoint times. Step zero has zero optimiser time. These timings describe the simulator and machine and are not end-to-end budgets or QPU measurements. No samples, Top-K repair, best-iterate selection or additional hyperparameter tuning occur.

Outputs include:

| File | Contents |
| --- | --- |
| `manifest.json` | Inherited settings and snapshots, checkpoint/draw allowances, both input-file hashes, software and source metadata |
| `trajectories.json` | Atomic checkpoint of completed trajectories, raw parameters/distributions at each step, metrics, errors and source-reproduction checks |
| `checkpoints.csv` | Per-run checkpoint metrics, including all three sampling allowances and cumulative optimisation times |
| `summary.csv` | Means over seeds within each instance and then equally across instances, with error and coverage counts |
| `paired_summary.csv` | Paired changes for 0→25, 0→100 and 25→100, including target probability, sampling success and additional training time |
| `report.md` | Descriptive results, definitions and limitations |
| `training_effort.png` | Mean target probability, feasibility, sampling reliability and training time |
| `training_pairs.png` | Individual paired target-probability trajectories and their means; each panel uses its own vertical scale |

Use `--resume` with identical inputs, settings, sources and recorded environment to skip completed trajectories. An interrupted trajectory restarts from its original seed; the runner does not resume Adam from a parameter snapshot without its moment estimates. Completed errors remain recorded. Failed trajectories have blank metrics at all checkpoints and are excluded from probability means, with coverage explicit. Only one process should write to an output directory.

There are no confidence intervals, noise studies, scaling conclusions or advantage claims. The analysis determines whether extra training helps under these fixed settings; a later comparison of chosen settings must use fresh evaluation instances. See the [saved training-effort report](../results/training_effort/report.md).

## Eight-security failure diagnostics

This development study investigates the three frozen eight-security problems rather than increasing the universe size. It combines exact XY reachability analysis with a declared depth/penalty grid: XY QAOA depths 1, 2 and 3 at the inherited penalty of 5; Binary VQE depths 2 and 3 at penalties 0.05, 0.5 and 5. All variants use the original three seeds, 100-step Adam settings and checkpoints 0, 25 and 100. Of 81 variant trajectories, 18 reuse the completed depth-1 XY and depth-2/alpha-5 VQE baselines; 63 require new training. Cached distributions are checked against the production circuit. Historical and new runtimes are not treated as a speed comparison.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_failure_diagnostics.py \
  --effort-dir results/training_effort \
  --assets 8 --xy-depths 1,2,3 --vqe-depths 2,3 \
  --penalties 0.05,0.5,5 --beta-points 257 \
  --output-dir results/my_failure_diagnostics
```

At XY depth 1, the first diagonal cost layer applies only a global phase to the basis initial state. The ordered exchange gates then have amplitudes `cos(2*beta)` for staying and `-i*sin(2*beta)` for swapping unequal bits; equal bits are unchanged. Writing `z=exp(2i*beta)` gives Laurent polynomials with integer coefficients and a shared power-of-two denominator. Exact integer cancellation determines which basis amplitudes can be nonzero for any beta. This proves support exclusion independently of a finite angle scan. At higher depth, gate-path enumeration supplies only an upper bound: cost layers cannot add support, but interference and tied angles can remove states. Inclusion in that bound is not proof of reachability. The first gamma is globally redundant at every depth; generic perturbation probes verify its numerical invariance and test later angles.

For each instance, exhaustive enumeration computes the strict sufficient penalty threshold `max(0, max_infeasible((f_star-f(x))/(sum(x)-K)^2))`. Alpha strictly above that threshold makes the global penalised optimum feasible, subject to floating-point evaluation; it does not guarantee optimisation success. A penalty below the threshold can change the penalised ground state to an infeasible portfolio. Every returned quality metric still uses the same original feasible-portfolio objective.

Binary VQE expressivity is checked separately: placing `pi*x` in the first RY layer and zeroing later rotations represents any basis portfolio, since CZ gates affect phases only. Saved target-state constructions are evaluation evidence and never initialise training. For each trained VQE mode the diagnostic records feasibility, objective gap, analytic gradient norm and the best one-for-one feasible swap. For improving swaps it calculates the maximum energy rise along a specified two-rotation product-state path, using a separate product-state construction of the modal bitstring rather than the trained parameter vector. That barrier illustrates one path, not every possible entangled path, and neither it nor a small gradient certifies a local optimum or a trapping mechanism.

Outputs include `static_analysis.json` (exact amplitude coefficients, support bounds and constructive VQE target parameters), support/parameter/beta-grid/penalty CSVs, raw `trajectories.json`, checkpoint and paired summary CSVs, and `vqe_modes.csv`. Three figures show support versus targets, the controlled sensitivity grid, and VQE convergence. The beta-grid maximum is exploratory rather than a certified global optimum; exact support exclusion is the basis for impossibility claims.

Use `--resume` to continue missing variants only when source hashes, settings and environment match. Finished errors remain recorded; interrupted trajectories restart from their seed with full Adam state reconstructed. Higher depth changes parameter dimension and the production random initialisation, so identical seed labels do not imply identical physical initial states. Results average seeds within each instance and then instances equally, retaining failure counts and coverage.

This is exploratory diagnosis on the existing development set, with no confidence intervals, held-out tuning, scaling claims or quantum-advantage claims. See the [saved report and figures](../results/failure_diagnostics/report.md).

## v0.3.1 reliability and execution-budget studies

```bash
python scripts/run_solver_reliability.py
python scripts/run_budget_benchmark.py --budgets 1,10,60
```

Both runners freeze synthetic problem snapshots, references, settings, software,
and recursive source hashes in `manifest.json`, checkpoint each trial, and accept
`--resume` only when code, environment, and settings still match. Use separate
output directories for different configurations.

The [reliability study](../results/solver_reliability/report.md) compares eight
Binary VQE and XY QAOA variants at equal total Adam updates, without claiming
equal runtime or circuit counts. The [deadline pilot](../results/budget_benchmark/report.md)
includes process startup and all candidate processing in each method's allowance;
late output is excluded and failures remain in the denominators. Independent
1/10/60-second runs retain incumbent trajectories and per-instance censored T90.
These are small development studies, not the full held-out research protocol.
See [usage](../USAGE.md) and [changelog](../CHANGELOG.md) for details.

## Fresh-instance deadline comparison

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_budget_benchmark.py \
  --config experiments/fresh_instance_budget.json \
  --output-dir results/fresh_instance_budget
```

The frozen configuration evaluates three fresh factor-model instances per size
(4, 6, and 8 assets), two run seeds, eleven named methods/variants across the two
objective tracks, and independent 1/10/60-second allowances: 594 trials total.
It rejects any problem ID present in the declared development manifests and
records their hashes. Settings are inherited from the development studies and
fixed before evaluating these fresh inputs; no tuning on the new results occurs.

The central binary comparison is trained versus untrained Dicke-ring XY QAOA
using the same production parameter initialization per seed. The untrained
control repeatedly executes actual finite-shot circuit batches without Adam
updates. Its trained counterpart publishes training checkpoints and samples its
final parameters for the remaining allowance after 24 updates. All three
sampling loops can stop after observing every feasible portfolio,
using the domain size rather than a reference optimum. Legacy XY and Binary VQE,
improved Binary VQE, QAOA X, HiGHS, swap search, and uniform feasible sampling
remain visible. Fractional VQE and OSQP form a separate allocation control.

Raw trials include independently validated incumbents and all failures.
`timings.csv` separates worker startup, setup, optimization, candidate processing,
delivery, validation, cleanup, and interrupted phases; delivery and validation
overlap worker phases. `pairs.csv` retains per-instance/seed/budget trained versus
comparator outcomes, including failed returns. The report shows descriptive
paired quality and target-hit results at the primary 10-second allowance.

Use `--resume` only with matching sources, environment, and settings. Historical
studies retain their original artifacts. This fresh-instance check has no
confidence intervals, formal tuning split, scaling claim, or advantage claim.
See [usage](../USAGE.md#fresh-instance-comparison-and-timing-diagnostics) for preset
semantics and [the saved report](../results/fresh_instance_budget/report.md).
