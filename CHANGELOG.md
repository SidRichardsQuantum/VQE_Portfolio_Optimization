# Changelog

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
