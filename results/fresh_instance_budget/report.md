# Fixed-deadline simulator pilot

Completed 594 of 594 trials. Every timeout and error remains in the denominators.

Spawn startup and all method-specific work consume the budget. Only portfolios received before the deadline are scored. Cleanup overruns are retained in the trial records. No shared warm worker or amortization is assumed.

| Assets | Variant | Seconds | Feasible | Target hit | Errors |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | binary-legacy | 1 | 0.0% | 0.0% | 0 |
| 4 | binary-legacy | 10 | 100.0% | 50.0% | 0 |
| 4 | binary-legacy | 60 | 100.0% | 50.0% | 0 |
| 4 | binary-improved | 1 | 0.0% | 0.0% | 0 |
| 4 | binary-improved | 10 | 100.0% | 83.3% | 0 |
| 4 | binary-improved | 60 | 100.0% | 83.3% | 0 |
| 4 | qaoa-x | 1 | 0.0% | 0.0% | 0 |
| 4 | qaoa-x | 10 | 100.0% | 100.0% | 0 |
| 4 | qaoa-x | 60 | 100.0% | 100.0% | 0 |
| 4 | xy-basis-ring | 1 | 0.0% | 0.0% | 0 |
| 4 | xy-basis-ring | 10 | 100.0% | 33.3% | 0 |
| 4 | xy-basis-ring | 60 | 100.0% | 33.3% | 0 |
| 4 | xy-dicke-ring-trained | 1 | 0.0% | 0.0% | 0 |
| 4 | xy-dicke-ring-trained | 10 | 100.0% | 100.0% | 0 |
| 4 | xy-dicke-ring-trained | 60 | 100.0% | 100.0% | 0 |
| 4 | xy-dicke-ring-untrained | 1 | 0.0% | 0.0% | 0 |
| 4 | xy-dicke-ring-untrained | 10 | 100.0% | 100.0% | 0 |
| 4 | xy-dicke-ring-untrained | 60 | 100.0% | 100.0% | 0 |
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
| 4 | fractional-vqe | 10 | 100.0% | 50.0% | 0 |
| 4 | fractional-vqe | 60 | 100.0% | 50.0% | 0 |
| 4 | osqp | 1 | 0.0% | 0.0% | 0 |
| 4 | osqp | 10 | 100.0% | 100.0% | 0 |
| 4 | osqp | 60 | 100.0% | 100.0% | 0 |
| 6 | binary-legacy | 1 | 0.0% | 0.0% | 0 |
| 6 | binary-legacy | 10 | 100.0% | 33.3% | 0 |
| 6 | binary-legacy | 60 | 100.0% | 33.3% | 0 |
| 6 | binary-improved | 1 | 0.0% | 0.0% | 0 |
| 6 | binary-improved | 10 | 100.0% | 100.0% | 0 |
| 6 | binary-improved | 60 | 100.0% | 100.0% | 0 |
| 6 | qaoa-x | 1 | 0.0% | 0.0% | 0 |
| 6 | qaoa-x | 10 | 100.0% | 100.0% | 0 |
| 6 | qaoa-x | 60 | 100.0% | 100.0% | 0 |
| 6 | xy-basis-ring | 1 | 0.0% | 0.0% | 0 |
| 6 | xy-basis-ring | 10 | 100.0% | 0.0% | 0 |
| 6 | xy-basis-ring | 60 | 100.0% | 0.0% | 0 |
| 6 | xy-dicke-ring-trained | 1 | 0.0% | 0.0% | 0 |
| 6 | xy-dicke-ring-trained | 10 | 100.0% | 100.0% | 0 |
| 6 | xy-dicke-ring-trained | 60 | 100.0% | 100.0% | 0 |
| 6 | xy-dicke-ring-untrained | 1 | 0.0% | 0.0% | 0 |
| 6 | xy-dicke-ring-untrained | 10 | 100.0% | 100.0% | 0 |
| 6 | xy-dicke-ring-untrained | 60 | 100.0% | 100.0% | 0 |
| 6 | highs | 1 | 0.0% | 0.0% | 0 |
| 6 | highs | 10 | 100.0% | 100.0% | 0 |
| 6 | highs | 60 | 100.0% | 100.0% | 0 |
| 6 | swap | 1 | 0.0% | 0.0% | 0 |
| 6 | swap | 10 | 100.0% | 100.0% | 0 |
| 6 | swap | 60 | 100.0% | 100.0% | 0 |
| 6 | random-k | 1 | 0.0% | 0.0% | 0 |
| 6 | random-k | 10 | 100.0% | 100.0% | 0 |
| 6 | random-k | 60 | 100.0% | 100.0% | 0 |
| 6 | fractional-vqe | 1 | 0.0% | 0.0% | 0 |
| 6 | fractional-vqe | 10 | 100.0% | 16.7% | 0 |
| 6 | fractional-vqe | 60 | 100.0% | 16.7% | 0 |
| 6 | osqp | 1 | 0.0% | 0.0% | 0 |
| 6 | osqp | 10 | 100.0% | 100.0% | 0 |
| 6 | osqp | 60 | 100.0% | 100.0% | 0 |
| 8 | binary-legacy | 1 | 0.0% | 0.0% | 0 |
| 8 | binary-legacy | 10 | 100.0% | 16.7% | 0 |
| 8 | binary-legacy | 60 | 100.0% | 16.7% | 0 |
| 8 | binary-improved | 1 | 0.0% | 0.0% | 0 |
| 8 | binary-improved | 10 | 100.0% | 33.3% | 0 |
| 8 | binary-improved | 60 | 100.0% | 33.3% | 0 |
| 8 | qaoa-x | 1 | 0.0% | 0.0% | 0 |
| 8 | qaoa-x | 10 | 100.0% | 100.0% | 0 |
| 8 | qaoa-x | 60 | 100.0% | 100.0% | 0 |
| 8 | xy-basis-ring | 1 | 0.0% | 0.0% | 0 |
| 8 | xy-basis-ring | 10 | 100.0% | 66.7% | 0 |
| 8 | xy-basis-ring | 60 | 100.0% | 66.7% | 0 |
| 8 | xy-dicke-ring-trained | 1 | 0.0% | 0.0% | 0 |
| 8 | xy-dicke-ring-trained | 10 | 100.0% | 100.0% | 0 |
| 8 | xy-dicke-ring-trained | 60 | 100.0% | 100.0% | 0 |
| 8 | xy-dicke-ring-untrained | 1 | 0.0% | 0.0% | 0 |
| 8 | xy-dicke-ring-untrained | 10 | 100.0% | 100.0% | 0 |
| 8 | xy-dicke-ring-untrained | 60 | 100.0% | 100.0% | 0 |
| 8 | highs | 1 | 0.0% | 0.0% | 0 |
| 8 | highs | 10 | 100.0% | 100.0% | 0 |
| 8 | highs | 60 | 100.0% | 100.0% | 0 |
| 8 | swap | 1 | 0.0% | 0.0% | 0 |
| 8 | swap | 10 | 100.0% | 100.0% | 0 |
| 8 | swap | 60 | 100.0% | 100.0% | 0 |
| 8 | random-k | 1 | 0.0% | 0.0% | 0 |
| 8 | random-k | 10 | 100.0% | 100.0% | 0 |
| 8 | random-k | 60 | 100.0% | 100.0% | 0 |
| 8 | fractional-vqe | 1 | 0.0% | 0.0% | 0 |
| 8 | fractional-vqe | 10 | 100.0% | 33.3% | 0 |
| 8 | fractional-vqe | 60 | 100.0% | 33.3% | 0 |
| 8 | osqp | 1 | 0.0% | 0.0% | 0 |
| 8 | osqp | 10 | 100.0% | 100.0% | 0 |
| 8 | osqp | 60 | 100.0% | 100.0% | 0 |

## Paired comparisons at the primary 10-second allowance

The trained variant is Dicke-ring XY QAOA. Positive mean normalized improvement favors training. Gap means use only pairs with two feasible returns; target-hit rates include every pair. Each row reports observed outcomes, without uncertainty estimates.

| Assets | Comparator | Pairs | Both feasible | Trained target hit | Comparator target hit | Mean improvement |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | binary-legacy | 6 | 6 | 100.0% | 50.0% | 0.0228055 |
| 4 | binary-improved | 6 | 6 | 100.0% | 83.3% | 0.00163562 |
| 4 | qaoa-x | 6 | 6 | 100.0% | 100.0% | 0 |
| 4 | xy-basis-ring | 6 | 6 | 100.0% | 33.3% | 0.0175813 |
| 4 | xy-dicke-ring-untrained | 6 | 6 | 100.0% | 100.0% | 0 |
| 4 | highs | 6 | 6 | 100.0% | 100.0% | 0 |
| 4 | swap | 6 | 6 | 100.0% | 100.0% | 0 |
| 4 | random-k | 6 | 6 | 100.0% | 100.0% | 0 |
| 6 | binary-legacy | 6 | 6 | 100.0% | 33.3% | 0.0319611 |
| 6 | binary-improved | 6 | 6 | 100.0% | 100.0% | 4.03998e-05 |
| 6 | qaoa-x | 6 | 6 | 100.0% | 100.0% | 0 |
| 6 | xy-basis-ring | 6 | 6 | 100.0% | 0.0% | 0.0512443 |
| 6 | xy-dicke-ring-untrained | 6 | 6 | 100.0% | 100.0% | 0 |
| 6 | highs | 6 | 6 | 100.0% | 100.0% | 0 |
| 6 | swap | 6 | 6 | 100.0% | 100.0% | 0 |
| 6 | random-k | 6 | 6 | 100.0% | 100.0% | 0 |
| 8 | binary-legacy | 6 | 6 | 100.0% | 16.7% | 0.0130497 |
| 8 | binary-improved | 6 | 6 | 100.0% | 33.3% | 0.00645618 |
| 8 | qaoa-x | 6 | 6 | 100.0% | 100.0% | 0.000147984 |
| 8 | xy-basis-ring | 6 | 6 | 100.0% | 66.7% | 0.00254305 |
| 8 | xy-dicke-ring-untrained | 6 | 6 | 100.0% | 100.0% | 0 |
| 8 | highs | 6 | 6 | 100.0% | 100.0% | 0 |
| 8 | swap | 6 | 6 | 100.0% | 100.0% | 0 |
| 8 | random-k | 6 | 6 | 100.0% | 100.0% | 0 |

## Timing diagnostics

Worker phase durations include the active phase up to termination. Startup includes spawn and imports. Missing startup completion is censored at the deadline. Delivery and parent validation overlap worker time and are not additive phases. See timings.csv for per-trial measurements and interrupted phases.

All durations below are seconds. Delivery is the mean over delivered incumbents, averaged over runs with delivery; a dash means none arrived. Parent validation and cleanup means include all runs.

| Variant | Budget | Startup | Setup | Optimization | Candidates | Delivery | Validation | Cleanup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| binary-legacy | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | 0.0077 |
| binary-legacy | 10 | 1.7476 | 0.0072 | 0.8286 | 0.1095 | 0.0004 | 0.0001 | 0.0612 |
| binary-legacy | 60 | 1.7009 | 0.0061 | 0.8279 | 0.1070 | 0.0004 | 0.0002 | 0.0612 |
| binary-improved | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | 0.0086 |
| binary-improved | 10 | 1.7047 | 0.0074 | 1.1637 | 0.1070 | 0.0003 | 0.0003 | 0.0616 |
| binary-improved | 60 | 1.6958 | 0.0073 | 1.2481 | 0.1196 | 0.0003 | 0.0002 | 0.0641 |
| qaoa-x | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | 0.0095 |
| qaoa-x | 10 | 1.8034 | 0.0071 | 1.1711 | 0.1721 | 0.0003 | 0.0002 | 0.0613 |
| qaoa-x | 60 | 1.7487 | 0.0066 | 1.1997 | 0.1865 | 0.0002 | 0.0002 | 0.0615 |
| xy-basis-ring | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | 0.0070 |
| xy-basis-ring | 10 | 1.7314 | 0.0074 | 1.3637 | 0.1895 | 0.0004 | 0.0001 | 0.0627 |
| xy-basis-ring | 60 | 1.8043 | 0.0068 | 1.3163 | 0.2064 | 0.0002 | 0.0002 | 0.0614 |
| xy-dicke-ring-trained | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | 0.0077 |
| xy-dicke-ring-trained | 10 | 1.7354 | 0.0116 | 1.4015 | 1.1535 | 0.0003 | 0.0001 | 0.0560 |
| xy-dicke-ring-trained | 60 | 1.7109 | 0.0132 | 1.3255 | 4.2725 | 0.0003 | 0.0001 | 0.0586 |
| xy-dicke-ring-untrained | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | 0.0076 |
| xy-dicke-ring-untrained | 10 | 1.7325 | 0.0068 | 0.0000 | 0.7837 | 0.0002 | 0.0001 | 0.0588 |
| xy-dicke-ring-untrained | 60 | 1.7425 | 0.0074 | 0.0000 | 1.5341 | 0.0004 | 0.0002 | 0.0616 |
| highs | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | 0.0077 |
| highs | 10 | 1.7191 | 0.0004 | 0.0161 | 0.0003 | 0.0001 | 0.0001 | 0.0626 |
| highs | 60 | 1.7227 | 0.0004 | 0.0163 | 0.0003 | 0.0004 | 0.0001 | 0.0623 |
| swap | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | 0.0078 |
| swap | 10 | 1.7887 | 0.0004 | 0.0049 | 0.0034 | 0.0003 | 0.0003 | 0.0610 |
| swap | 60 | 1.7870 | 0.0004 | 0.0040 | 0.0030 | 0.0003 | 0.0002 | 0.0612 |
| random-k | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | 0.0081 |
| random-k | 10 | 1.8294 | 0.0005 | 0.0000 | 0.0095 | 0.0003 | 0.0003 | 0.0609 |
| random-k | 60 | 1.8686 | 0.0006 | 0.0000 | 0.0083 | 0.0001 | 0.0004 | 0.0612 |
| fractional-vqe | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | 0.0084 |
| fractional-vqe | 10 | 1.8471 | 0.0014 | 0.2427 | 0.0323 | 0.0003 | 0.0004 | 0.0622 |
| fractional-vqe | 60 | 1.8764 | 0.0012 | 0.2246 | 0.0283 | 0.0003 | 0.0005 | 0.0608 |
| osqp | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | — | 0.0000 | 0.0089 |
| osqp | 10 | 1.8153 | 0.0004 | 0.1829 | 0.0002 | 0.0003 | 0.0001 | 0.0635 |
| osqp | 60 | 1.8275 | 0.0004 | 0.1703 | 0.0002 | 0.0004 | 0.0001 | 0.0625 |

## Fresh-instance and sampling controls

Excluded development manifests and their hashes are frozen in manifest.json; overlap is rejected before solving. Named presets freeze the v0.3.1 reliability settings. Trained/untrained Dicke-ring variants share production parameter initialization for each seed. The trained variant publishes candidates during training, then spends the remaining allowance repeatedly sampling its final parameters. The untrained variant repeatedly executes the initial circuit with zero Adam updates and actual finite-shot batches. Uniform sampling and both circuit sampling loops may stop once every feasible exact-K portfolio has actually been sampled, using only n and K, never a reference target. Otherwise sampling uses the full allowance. Finite training caps and early completion are explicit algorithm settings; equal allowances do not imply equal time spent.

pairs.csv compares the trained Dicke-ring variant with each binary comparator on identical instance/seed/budget keys. Positive normalized improvement favors training. Quality differences require both feasible returns; failure and target-hit flags remain explicit. These are descriptive pairs, without confidence intervals or a formal held-out tuning study.

Quality means condition on feasible returns, with coverage in summary.csv. reliability.csv reports per-instance empirical T90 using all runs, including censored failures; a blank means unreached within the budget. Reference computation is excluded. Binary and fractional objectives remain separate. These are descriptive simulator results, without confidence intervals, held-out tuning, a scaling claim, or hardware advantage evidence.
