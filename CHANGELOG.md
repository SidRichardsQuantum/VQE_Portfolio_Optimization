# Changelog

## v0.3.5

- Add opt-in lower-tail CVaR training to Binary VQE and both QAOA mixers,
  including lambda sweeps, restart selection, normalization, best-iterate
  selection and solver CLI/data commands. Preserve expectation defaults and
  the expectation path at a full tail fraction of 1.
- Use differentiable state probabilities with Adam for analytic CVaR, and
  empirical fractional-boundary tails with seeded SPSA for finite-shot CVaR.
  Record the objective, tail fraction and optimizer in solver results.

## v0.3.4

- Add individual quantum and classical algorithm pages, shared examples, and
  documentation navigation under `docs/algorithms/`.
- Fix top-return selection at K=0 and validate its cardinality through the common
  problem definition. Add boundary and invalid-input regression coverage.
- Correct allocator normalisation to `(s+epsilon/n)/(sum(s)+epsilon)`, including
  zero measured mass, and test analytic/finite-difference gradients and shots.
- Introduce `run_variational_allocator`, allocator config/result/sweep names,
  `allocator` CLI commands, and `variational-allocator` study IDs. Retain the
  fractional-VQE names as compatibility aliases and preserve historical outputs.
- Clarify Hamiltonian VQE versus the custom allocator and ordered XY mixing;
  test ideal exact-cardinality preservation and the exchange-gate product.
- Save a fresh 128-trial correctness comparison across 4/6-asset binary and
  fractional problems with source/environment provenance; all returns are
  feasible, with zero solver errors or reference contradictions. Preserve
  historical artifacts and distinguish target quality from feasibility.
- Add NumPy-only memetic genetic search, iterated local search (ILS), and GRASP
  for the existing exact-K binary objective. Reuse best-improvement swap
  refinement and publish copied, improving feasible incumbents.
- Expose `memetic`, `ils`, and `grasp` in comparison CSVs, small-instance
  validation, and the budget CLI/runner; include all three in `classical-solvers`.
- Add `qp-round`: solve the exact-K box QP relaxation with CVXPY/OSQP, round
  deterministically, then refine using feasible swaps. Preserve the original
  binary objective and publish copied improving incumbents; numerical
  relaxation diagnostics are separate from optimality bounds.
- Add NumPy-only VNS for exact-K binary selection and projected gradient
  descent for the fractional simplex objective. Expose `vns` and `pgd` in
  comparisons, validation, and execution budgets, with feasible incumbent
  callbacks and explicit iteration/numerical stopping diagnostics.

## v0.3.3

- Add NumPy-only simulated annealing and tabu search for exact-K binary
  selection, with seeded restarts, best-feasible incumbents, and callbacks.
- Expose `anneal` and `tabu` in synthetic comparisons, small-instance validation,
  and the budget CLI/runner. Include both in `classical-solvers`; allow individual
  solver selection without loading unselected optional solvers.
- Add a SCIP quadratic integer baseline with an unrestricted objective epigraph,
  exact cardinality, native status/bound/gap metadata, and streaming feasible
  incumbents. Expose `scip` in comparisons, validation, and execution budgets.
- Include PySCIPOpt in the `classical` and `all` extras, add a standalone `scip`
  extra, and record its version in new experiment manifests.

## v0.3.2

- Add fresh-instance deadline comparisons of trained and untrained Dicke-ring
  XY QAOA, legacy and improved Binary VQE, and classical comparators. Freeze
  named presets and reject overlap with declared development manifests.
- Record startup, setup, optimization, candidate processing, delivery,
  parent validation, and cleanup timing diagnostics. Parent validation must
  finish before the startup-inclusive deadline.
- Add repeated sampling at initial and trained parameters, and reference-free stopping
  after every feasible exact-K portfolio has actually been sampled. QAOA also
  accepts zero training steps for initial-state inspection.
- Require PennyLane 0.42.3 or newer for the solver shot-setting API and
  setuptools 77 or newer for SPDX license metadata. Test minimum core
  dependencies alongside current dependencies on Python 3.10–3.12.
- Gate PyPI publishing on the complete CI workflow and require a release tag
  matching the package version. Check built artifact metadata before upload.

## v0.3.1

- Add opt-in uniform exact-K initialization and complete exchange networks for
  XY QAOA, preserving the legacy basis/ring defaults and lambda-sweep support.
- Add reference-free auto penalties, energy normalization, multiple restarts,
  best-iterate selection, and learning-rate decay for Binary VQE. Record resolved
  penalties, normalization scales, and restart scores.
- Align optimizer trace values with post-update parameter snapshots and always
  record the final update. Logging now performs a matching cost evaluation;
  finite-shot random streams and runtimes can change.
- Add isolated-process end-to-end execution budgets, timestamped feasible
  incumbents, independent scoring, explicit failures, and common termination
  metadata for quantum and classical methods. Include startup and candidate
  processing; reject late output and report cleanup overruns.
- Expose reliability controls and budgets in the CLI. Add resumable reliability
  and fixed-budget experiment runners with frozen inputs and source provenance.
- Preserve historical experiment artifacts. New development reports compare
  algorithm settings and document limits on performance claims.
