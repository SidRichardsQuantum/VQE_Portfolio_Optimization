# Changelog

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
