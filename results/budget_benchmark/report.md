# Fixed-deadline simulator pilot

Completed 24 of 24 trials. Every timeout and error remains in the denominators.

Spawn startup and all method-specific work consume the budget. Only portfolios received before the deadline are scored. Cleanup overruns are retained in the trial records. No shared warm worker or amortization is assumed.

| Assets | Method | Seconds | Feasible | Target hit | Errors |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | binary-vqe | 1 | 0.0% | 0.0% | 0 |
| 4 | binary-vqe | 10 | 100.0% | 0.0% | 0 |
| 4 | binary-vqe | 60 | 100.0% | 0.0% | 0 |
| 4 | qaoa-x | 1 | 0.0% | 0.0% | 0 |
| 4 | qaoa-x | 10 | 100.0% | 100.0% | 0 |
| 4 | qaoa-x | 60 | 100.0% | 100.0% | 0 |
| 4 | qaoa-xy | 1 | 0.0% | 0.0% | 0 |
| 4 | qaoa-xy | 10 | 100.0% | 0.0% | 0 |
| 4 | qaoa-xy | 60 | 100.0% | 0.0% | 0 |
| 4 | highs | 1 | 0.0% | 0.0% | 0 |
| 4 | highs | 10 | 100.0% | 100.0% | 0 |
| 4 | highs | 60 | 100.0% | 100.0% | 0 |
| 4 | swap | 1 | 0.0% | 0.0% | 0 |
| 4 | swap | 10 | 100.0% | 100.0% | 0 |
| 4 | swap | 60 | 100.0% | 100.0% | 0 |
| 4 | random-k | 1 | 0.0% | 0.0% | 0 |
| 4 | random-k | 10 | 100.0% | 100.0% | 0 |
| 4 | random-k | 60 | 100.0% | 100.0% | 0 |
| 4 | fractional-vqe | 1 | 0.0% | 0.0% | 0 |
| 4 | fractional-vqe | 10 | 100.0% | 100.0% | 0 |
| 4 | fractional-vqe | 60 | 100.0% | 100.0% | 0 |
| 4 | osqp | 1 | 0.0% | 0.0% | 0 |
| 4 | osqp | 10 | 100.0% | 100.0% | 0 |
| 4 | osqp | 60 | 100.0% | 100.0% | 0 |

Quality means condition on feasible returns, with coverage in summary.csv. reliability.csv reports per-instance empirical T90 using all runs, including censored failures; a blank means unreached within the budget. Reference computation is excluded. Binary and fractional objectives remain separate. These are descriptive simulator results, without confidence intervals, held-out tuning, a scaling claim, or hardware advantage evidence.
