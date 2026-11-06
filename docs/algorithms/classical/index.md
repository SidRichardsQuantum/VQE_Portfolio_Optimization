# Classical algorithms

These pages cover the implemented solver wrappers, NumPy searches, and simple
reference baselines. Read the [shared objectives and inputs](../index.md) first.
Entries without a study method ID remain available through the Python API.

## Exact references and optimization backends

- [Exhaustive binary search](exhaustive-search.md)
- [Markowitz support enumeration](exact-markowitz.md)
- [Convex Markowitz QP with OSQP](osqp.md)
- [Binary MILP with HiGHS](highs.md)
- [Binary quadratic optimization with SCIP](scip.md)

## Search and relaxation methods

- [QP relaxation, rounding, and refinement](qp-rounding.md)
- [Multistart swap local search](swap-search.md)
- [Simulated annealing](simulated-annealing.md)
- [Tabu search](tabu-search.md)
- [Memetic genetic algorithm](memetic-genetic.md)
- [Iterated local search](iterated-local-search.md)
- [GRASP](grasp.md)
- [Variable neighborhood search](variable-neighborhood-search.md)
- [Projected gradient descent](projected-gradient-descent.md)
- [Frank–Wolfe with away steps](frank-wolfe.md)

## Simple comparison baselines

- [Top-return selection](top-return.md)
- [Minimum-variance binary selection](minimum-variance.md)
- [Equal-weight allocation](equal-weight.md)
- [Random exact-K sampling](random-k.md)

```{toctree}
:hidden:
:maxdepth: 1

exhaustive-search
top-return
minimum-variance
exact-markowitz
equal-weight
osqp
highs
scip
qp-rounding
swap-search
simulated-annealing
tabu-search
memetic-genetic
iterated-local-search
grasp
variable-neighborhood-search
projected-gradient-descent
frank-wolfe
random-k
```
