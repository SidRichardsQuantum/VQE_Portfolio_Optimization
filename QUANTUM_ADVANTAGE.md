# Quantum Advantage Research Protocol

Status: prospective protocol. The existing results are exploratory examples, not evidence that these criteria have been met. This document defines the research question; the full fixed-budget study remains subsequent work. Shared problem snapshots, independent candidate evaluation, and stronger classical baselines are implemented; see [usage](USAGE.md#shared-problem-definitions-and-evaluation).

## Research question

For a specified portfolio problem family, can Binary VQE, QAOA, or Fractional VQE return a feasible portfolio of a specified quality faster than strong classical methods, or return better quality within the same computational budget?

An advantage claim must name the formulation, instance distribution, size range, solvers, hardware, and resource budget tested. Failure to find an advantage does not prove that no future quantum algorithm could have one.

## Problems being compared

Keep two independent tracks matching the current package objectives:

| Track | Objective to minimise | Feasible portfolios |
| --- | --- | --- |
| Binary selection | `f(x) = lambda * x.T @ Sigma @ x - mu.T @ x` | Binary `x`, with `sum(x) = K` |
| Fractional allocation | `f(w) = lambda * w.T @ Sigma @ w - mu.T @ w` | `w >= 0`, with `sum(w) = 1` |

All methods in a track receive identical frozen `mu`, `Sigma`, risk aversion, and constraints. Classical methods may use their natural constrained formulation. Quantum penalty coefficients are algorithm settings, not changes to the underlying comparison problem.

Check binary feasibility exactly. For fractional output, require finite entries, `min(w) >= -1e-8`, and `abs(sum(w) - 1) <= 1e-8`. Any repair or renormalisation is part of the algorithm and must be timed and reported. Score the returned feasible candidate, not an expectation over infeasible candidates. Runs returning no feasible candidate are failures.

The binary objective uses raw selections `x`, as in the existing implementation. Risk and return reported using `x / K` are descriptive metrics; they must not replace that objective or be compared with fractional objectives as if the problems were identical.

## Quality target

For each instance, obtain a certified optimum `f_star` independently where practical. Otherwise retain a valid lower bound `L` and the best known feasible objective `U`, explicitly identifying the unresolved interval. Reference computation is benchmark preparation, not information available to the algorithms.

Use a fixed, positive, instance-specific scale:

- Binary: `S = abs(lambda) * K**2 * max(abs(Sigma)) + K * max(abs(mu))`.
- Fractional: `S = abs(lambda) * max(abs(Sigma)) + max(abs(mu))`.

Exclude instances with `S = 0` from advantage claims because every feasible portfolio has the same objective. Define the normalised gap as `(f - f_star) / S`. This avoids division by a zero or negative optimum. Always report the raw objective and absolute gap too; the normalised gap is not a percentage investment loss.

The default target is a feasible solution with gap at most `1e-3`. Where only bounds are available, `(f - L) / S <= 1e-3` certifies the target. Reaching `U + 1e-3 * S` only establishes a best-known-reference target, not near-optimality. Report that result separately. Freeze references before evaluation; investigate any candidate that contradicts a certified optimum.

## Budgets and outcomes

Default per-run end-to-end budgets are **1, 10, and 60 seconds**, with **10 seconds** the primary fixed-budget endpoint. These are research defaults, not established financial requirements. Any revised budget or tolerance must be recorded before held-out evaluation.

Start the clock when a method receives the common prepared instance. Include method-specific preprocessing, circuit compilation, parameter optimisation, sampling, communication, mitigation, retries, and postprocessing. Stop when a usable portfolio is available. Report service queue time separately and also in user-observed elapsed time. Record CPU/GPU/QPU resources and monetary cost separately; equal elapsed time alone does not establish cost advantage. Report tuning costs and any amortisation assumptions.

Record two outcomes:

1. **Primary: quality at a fixed budget.** At 10 seconds, compare the best feasible objective returned within the deadline. Report feasibility and target-hit rates alongside gap distributions. Never silently discard failed runs or average them as successful portfolios.
2. **Secondary: time to target.** Record the first time a feasible target-quality portfolio is available. Report the empirical 90th-percentile time across independent runs. If fewer than 90% reach the target within 60 seconds, report it as unreached within budget; do not compute a success-only timing average or extrapolate it.

## Decision rules

Before evaluation, freeze an experiment manifest containing instance IDs and data hashes, size groups, reference bounds, solver versions, settings, seeds, hardware, budgets, and tuning/evaluation split. Use at least 30 held-out instances per declared size group and 30 independent runs per stochastic method and instance. Repeated seeds do not substitute for independent instances. Repeat deterministic timings to quantify runtime variability.

Compare against efficient convex optimisation for the fractional track and strong constrained integer optimisation and heuristics for the binary track. Enumeration remains a small-instance correctness reference. Tune competitors with comparable effort on separate training instances. Existing repo baselines alone are insufficient for a quantum-advantage claim.

- **Quality win:** at the primary budget, the lower endpoint of a paired 95% confidence interval for the mean normalised improvement `(f_classical - f_quantum) / S` exceeds `1e-3`, with no observed reduction in feasibility or target-hit rate. Compute paired quality differences only where both methods return feasible candidates and publish the paired coverage and all failures. If either method has failures, treat the quality comparison as conditional and require a separately preregistered failure-aware analysis before claiming an overall win.
- **Speed win:** both methods reach the same certified target with at least 90% empirical reliability, and the lower endpoint of a paired 95% confidence interval for the speed ratio `T90_classical / T90_quantum` exceeds `1.1`.
- Resample instances as the outer unit and runs within instances when estimating uncertainty. Correct for multiple comparisons across methods, size groups, and claimed endpoints; identify secondary analyses as exploratory unless covered by that correction. Report uncertainty in success rates as well as objectives and times.
- A qualifying result must hold against every declared strong classical comparator in the claimed group, not just the weakest baseline. Publish the full group, including losses. These thresholds are practical research choices and must not be adjusted after seeing evaluation results.

## Interpretation

Simulator results establish behaviour of the simulated implementation, not quantum hardware speed advantage. Hardware projections are resource estimates, not measured wins. A practical quantum-advantage claim requires measured quantum hardware performance and end-to-end classical comparisons under the rules above. A finite-size win does not establish an asymptotic speedup.

Use conclusions such as: “No advantage observed for these methods, instances, budgets, and hardware,” or “Measured advantage over the specified classical baselines in this size range.” Avoid universal claims about all classical algorithms or all portfolio problems.

Future realised returns, Sharpe ratios, and backtests address a separate investment-performance question. Better historical returns alone do not demonstrate computational advantage, and a better optimisation objective does not guarantee better future returns.

## Methodological references

- [Towards Robust Benchmarking of Quantum Optimization Algorithms](https://arxiv.org/abs/2405.07624): appropriate formulations, comparable tuning effort, and time/quality metrics.
- [Quantum Portfolio Optimization: An Extensive Benchmark](https://arxiv.org/abs/2509.17876): portfolio-specific evidence motivating strong classical comparators.

The numerical thresholds and decision rules above are this project's prospective choices, not claims established by these references.
