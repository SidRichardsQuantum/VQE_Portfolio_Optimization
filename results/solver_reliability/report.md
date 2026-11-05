# Solver reliability development study

Completed 216 of 216 planned trials.

Binary variants compare the legacy penalty, a reference-free sufficient auto penalty, normalization, and a combined restart/best-iterate/learning-rate-decay strategy. Each variant receives the same total number of Adam updates; restarts divide those updates equally. They do not have equal wall-clock or circuit-evaluation budgets. XY variants compare basis and uniform exact-K (Dicke) initial states with ring and complete ordered exchange networks at depth 1.

All settings and problem references are frozen in manifest.json. Optimizers never receive reference scores or target sets. Exact noiseless target probabilities are evaluated after solving; they are distinct from the sampled candidate hit rate. Success at m draws is computed per run before averaging seeds within instances and giving instances equal weight.

| Assets | Variant | Target per draw | Feasible | Success: 64 draws | Uniform exact-K: 64 draws | Errors |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | binary-legacy | 26.551% | 97.311% | 44.583% | 99.999% | 0 |
| 4 | binary-auto | 53.160% | 95.866% | 70.910% | 99.999% | 0 |
| 4 | binary-auto-normalized | 53.160% | 95.866% | 70.910% | 99.999% | 0 |
| 4 | binary-restarts | 51.374% | 95.960% | 57.350% | 99.999% | 0 |
| 4 | xy-basis-ring | 41.138% | 100.000% | 66.667% | 99.999% | 0 |
| 4 | xy-dicke-ring | 33.456% | 100.000% | 100.000% | 99.999% | 0 |
| 4 | xy-basis-complete | 36.111% | 100.000% | 67.493% | 99.999% | 0 |
| 4 | xy-dicke-complete | 40.546% | 100.000% | 99.992% | 99.999% | 0 |
| 6 | binary-legacy | 18.472% | 97.536% | 27.180% | 97.459% | 0 |
| 6 | binary-auto | 52.224% | 96.492% | 55.586% | 97.459% | 0 |
| 6 | binary-auto-normalized | 52.224% | 96.492% | 55.586% | 97.459% | 0 |
| 6 | binary-restarts | 45.850% | 85.501% | 60.071% | 97.459% | 0 |
| 6 | xy-basis-ring | 34.962% | 100.000% | 66.615% | 97.459% | 0 |
| 6 | xy-dicke-ring | 17.715% | 100.000% | 99.794% | 97.459% | 0 |
| 6 | xy-basis-complete | 1.128% | 100.000% | 31.322% | 97.459% | 0 |
| 6 | xy-dicke-complete | 24.858% | 100.000% | 99.191% | 97.459% | 0 |
| 8 | binary-legacy | 0.003% | 96.702% | 0.187% | 68.241% | 0 |
| 8 | binary-auto | 0.061% | 96.339% | 3.295% | 68.241% | 0 |
| 8 | binary-auto-normalized | 0.061% | 96.339% | 3.292% | 68.241% | 0 |
| 8 | binary-restarts | 3.678% | 81.419% | 11.885% | 68.241% | 0 |
| 8 | xy-basis-ring | 0.000% | 100.000% | 0.000% | 68.241% | 0 |
| 8 | xy-dicke-ring | 4.331% | 100.000% | 90.958% | 68.241% | 0 |
| 8 | xy-basis-complete | 0.183% | 100.000% | 9.905% | 68.241% | 0 |
| 8 | xy-dicke-complete | 2.067% | 100.000% | 57.538% | 68.241% | 0 |

Errors remain in trial counts; probability means condition on successful runs and report instance coverage. The restart variant changes several controls together, so its outcome does not identify the contribution of any one control. StatePrep uses an explicit 2**n amplitude vector for the Dicke state; this is a small-instance simulator implementation, without a hardware preparation-cost claim. Complete networks use O(n**2) exchange edges. These are development instances with no held-out tuning, confidence intervals, scaling law, or quantum-advantage claim. Historical evidence is unchanged.
