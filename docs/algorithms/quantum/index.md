# Quantum algorithms

The package implements Hamiltonian-based Binary VQE, Filtering VQE, and QAOA, plus a custom
variational circuit allocator. All default to quantum simulation. Read the
[shared objectives and inputs](../index.md) before running the examples.

- [Binary VQE](binary-vqe.md): penalized Hamiltonian training and sampled selection.
- [Filtering VQE](filtering-vqe.md): normalized filtered-state distance updates,
  with inverse or exponential filters and analytic or finite-shot execution.
- [QAOA with X mixing](qaoa-x.md): cost phases and unconstrained bit flips.
- [QAOA with XY exchange mixing](qaoa-xy.md): ideal cardinality-preserving exchanges.
- [CVaR training](cvar.md): optional lower-tail cost aggregation for Binary VQE
  and QAOA, using analytic Adam or finite-shot SPSA.
- [Variational quantum allocator](variational-allocator.md): circuit expectations
  mapped to simplex weights, with a classically evaluated objective.

Circuit ansatz choices, mixer topologies, initialization choices, and trained
versus untrained presets are settings of these algorithms. Compatibility aliases
do not identify additional implementations.

```{toctree}
:hidden:
:maxdepth: 1

binary-vqe
filtering-vqe
qaoa-x
qaoa-xy
cvar
variational-allocator
```
