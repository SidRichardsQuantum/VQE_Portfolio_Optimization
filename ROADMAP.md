# Roadmap

Planned algorithm implementations for reproducible portfolio optimization and
classical–quantum comparisons. This is a development list, without release or
date commitments. Checked entries are implemented; unchecked entries are planned
work, not available features. See the [algorithm guides](docs/algorithms/index.md)
for current implementations.

## Scope and implementation requirements

All additions must fit the package's existing mean–variance objectives:

- **Binary selection:** minimize `lam * x.T @ Sigma @ x - mu.T @ x`, with
  binary `x` and exactly `K` selected assets.
- **Fractional allocation:** minimize `lam * w.T @ Sigma @ w - mu.T @ w`, with
  nonnegative weights summing to one. Convex methods require PSD covariance.

Binary selection scores raw selections; equal-weight reporting uses `x / K`
when `K > 0`. Fractional and binary scores remain separate. See the
[theory](THEORY.md) and [architecture](docs/architecture.md).

A true implementation must execute the defining mathematical procedure of its
named algorithm. Cite the method and document any adaptations. Distinguish
algorithm variants, optimizer settings, and compatibility aliases from new
algorithm families. Optional solver backends must solve the stated formulation.

For every addition:

- Integrate the Python API, relevant CLI and comparison runners, independent
  candidate evaluation, incumbent callbacks, and end-to-end execution budgets.
- Preserve reproducible seeds and report configuration, termination status,
  feasibility, and the original objective separately from training penalties.
- Test the defining algorithmic operations and small instances against exact
  references. Optimizers must not consult reference optima during execution.
- Account for preprocessing, search or training, candidate extraction, and
  relevant circuit, shot, oracle, or solver costs in comparisons.
- Document dependency requirements, numerical limitations, and supported
  analytic or finite-shot execution. Simulator results alone do not establish
  quantum advantage.

## Classical algorithms

- [x] **Frank–Wolfe with away steps — fractional allocation**
  - Implemented as `frank_wolfe_baseline` with method ID `frank-wolfe`.
    See the [algorithm guide](docs/algorithms/classical/frank-wolfe.md).
  - Implements the simplex linear minimization oracle, active-set maintenance,
    forward and away directions, feasible step bounds, and quadratic line search.
  - Reports a first-order gap and explicit stopping conditions. Tests compare
    with PGD and OSQP, including boundary solutions and singular PSD covariance.
  - Uses NumPy only. Sparse intermediate allocations do not constitute an
    exact-cardinality guarantee.
  - Reference: [Lacoste-Julien and Jaggi, Frank–Wolfe variants](https://papers.neurips.cc/paper_files/paper/2015/hash/c058f544c737782deacefa532d9add4c-Abstract.html).

- [x] **Parallel tempering / replica exchange — binary selection**
  - Implemented as `parallel_tempering_binary_baseline` with method ID
    `parallel-tempering`. See the [algorithm guide](docs/algorithms/classical/parallel-tempering.md).
  - Maintains replicas at distinct temperatures, uses exact-`K` swap proposals
    within replicas, and implements temperature-dependent Metropolis acceptance.
  - Exchanges complete replica states using the replica-exchange acceptance
    rule and retains the best feasible portfolio across all replicas.
  - Uses quadratic swap calculations and records the temperature ladder, local
    and exchange acceptance statistics, and total work.
    Tests include detailed balance, cache consistency, and a comparison with
    simulated annealing on a swap-local-minimum instance.
  - Reference: [Hukushima and Nemoto, exchange Monte Carlo](https://arxiv.org/abs/cond-mat/9512035).

- [ ] **Cross-entropy method / estimation of distribution — binary selection**
  - Implement repeated population sampling, elite selection, and distribution
    fitting, with explicit smoothing and controls for premature collapse.
  - Choose a documented exact-`K` sampling distribution and fit that same model.
    For example, conditioning Bernoulli variables on cardinality requires the
    conditional likelihood; independent elite-frequency updates are not generally
    its maximum-likelihood fit.
  - Report population size, elite fraction, fitting work, and sampled objective
    evaluations. Identify optional local refinement separately from distribution
    learning, and compare with random-`K` sampling and memetic genetic search.
  - Reference: [de Boer et al., cross-entropy tutorial](https://people.smp.uq.edu.au/DirkKroese/ps/CEtutfinal.pdf).

- [ ] **Semidefinite relaxation with rounding — binary selection**
  - Derive a lifted matrix relaxation of the existing exact-`K` binary objective,
    retaining valid binary and cardinality constraints while relaxing rank.
  - Use an optional SDP-capable solver, then construct exact-`K` candidates with
    documented rounding and optional swap refinement. OSQP is not an SDP solver.
  - Report relaxation results and feasible candidate objectives separately.
    Treat numerical dual bounds according to solver tolerances and validation;
    do not present an approximate primal relaxation value as a certified bound.
  - Reference: [Wiegele and Zhao, cardinality-constrained SDP relaxations](https://arxiv.org/abs/2107.11338).
    Adapt the formulation to this package's binary selection model rather than
    substituting a different sparse-weight portfolio problem.

## Quantum algorithms

- [ ] **Variational quantum imaginary-time evolution (VarQITE) — binary selection**
  - Implement McLachlan's variational imaginary-time equations using the circuit
    derivative overlap matrix and Hamiltonian derivative terms, solve for the
    parameter velocity, and integrate the resulting updates.
  - Reuse the portfolio Hamiltonian and suitable ansätze. Specify phase handling,
    regularization, linear-solve tolerances, and integration controls; test the
    evolution equations and behavior under singular overlap matrices.
  - Keep the procedure distinct from ordinary energy-gradient optimization and
    existing F-VQE filtering updates. Account for overlap and derivative
    evaluations, and document whether they use simulator derivatives or
    measurement circuits.
  - Reference: [McArdle et al., variational imaginary-time simulation](https://www.nature.com/articles/s41534-019-0187-2).

- [ ] **ADAPT-QAOA — binary selection**
  - Define an operator pool, evaluate candidate energy derivatives, select mixer
    operators, grow the alternating circuit, and reoptimize accumulated angles.
  - Document the initial state, cost-phase initialization, pool, selection rule,
    and stopping criteria. A fixed circuit with a different depth is insufficient.
  - Count operator screening and reoptimization toward budgets. Test selection
    gradients and circuit growth; if offering a cardinality-preserving variant,
    verify both state preparation and every selected mixer preserve exact `K`.
  - Reference: [Zhu et al., adaptive QAOA](https://arxiv.org/abs/2005.10258).
    Document this as a QAOA variant with an adaptive construction procedure.

- [ ] **Digitized quantum annealing / adiabatic optimization — binary selection**
  - Prepare the chosen driver ground state and implement a scheduled interpolation
    to the portfolio Hamiltonian using a documented product-formula circuit.
  - Expose total evolution time, schedule, discretization, and penalty choices.
    Derive gate angles from the schedule; distinguish this execution procedure
    from QAOA with freely optimized angles despite shared circuit components.
  - Test short-time evolution and discretization against tiny exact simulations.
    Report simulated evolution time separately from wall-clock cost and circuit
    depth. Finite-time execution does not guarantee an optimum.
  - Reference: [Farhi et al., quantum adiabatic evolution](https://arxiv.org/abs/quant-ph/0001106).

- [ ] **Grover adaptive search — binary selection**
  - Implement reversible objective encoding, threshold comparison, amplitude
    amplification, measurement, and classical threshold updates, with explicit
    cardinality enforcement or a justified penalty formulation.
  - Specify fixed-point coefficient precision, arithmetic range, ancilla use,
    uncomputation, and the effect of quantization on the original objective.
  - Construct the oracle from the problem coefficients. Exhaustively listing
    improving portfolios to build a lookup oracle is insufficient for the planned
    implementation. Test arithmetic and oracle behavior on tiny instances.
  - Report oracle calls and their gate and qubit costs, including construction
    overhead. Start with small simulator studies and make no speedup claim from
    oracle-query counts alone.
  - Reference: [Gilliam et al., Grover adaptive search and portfolio optimization](https://quantum-journal.org/papers/q-2021-04-08-428/).

## Suggested implementation order

Frank–Wolfe and parallel tempering are implemented. Continue with VarQITE and
ADAPT-QAOA. Follow with cross-entropy search and digitized annealing. SDP
relaxation and Grover adaptive search require additional solver or arithmetic infrastructure
and are larger projects. This ordering is a priority suggestion, not a release
schedule.
