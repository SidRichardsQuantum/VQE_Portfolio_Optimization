# Small-instance validation pilot

Completed 198 of 198 planned trials.

Synthetic factor-model instances; noiseless default.qubit simulator; exact-expectation training and finite-shot binary sampling.

This is a correctness/reliability pilot, not a fixed-budget study or a quantum-advantage test. Timings describe this simulator and machine only. Reference preparation is excluded from solver runtimes. Deterministic solvers run once per instance; stochastic methods use every configured solver seed.

Target: feasible normalised objective gap <= the configured target_gap (default 0.001). Gap scale is fixed per instance, not a percentage investment loss. Binary and fractional objectives are separate tracks. Fractional references are numerical active-set solutions, not exact-arithmetic certificates.

Rates and runtimes average instance means. Quality averages feasible-run means within each instance and excludes instances without a feasible return; inspect quality_instance_count and all trial failures. No confidence intervals or significance claims are made for this pilot.

| Assets | K | Method | Instances | Trials | Feasible | Target hit | Mean gap (feasible) | Mean seconds | Errors | Reference contradictions |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 | exact-binary | 3 | 3 | 100.0% | 100.0% | 0 | 0.000102294 | 0 | 0 |
| 4 | 2 | highs | 3 | 3 | 100.0% | 100.0% | 0 | 0.0118016 | 0 | 0 |
| 4 | 2 | swap | 3 | 9 | 100.0% | 100.0% | 0 | 0.00225046 | 0 | 0 |
| 4 | 2 | random-k | 3 | 9 | 100.0% | 100.0% | 0 | 0.00536162 | 0 | 0 |
| 4 | 2 | binary-vqe | 3 | 9 | 100.0% | 66.7% | 0.00970532 | 0.479065 | 0 | 0 |
| 4 | 2 | qaoa-x | 3 | 9 | 100.0% | 100.0% | 0 | 0.6041 | 0 | 0 |
| 4 | 2 | qaoa-xy | 3 | 9 | 100.0% | 66.7% | 0.00895049 | 0.782603 | 0 | 0 |
| 4 | — | exact-fractional | 3 | 3 | 100.0% | 100.0% | 0 | 0.00126936 | 0 | 0 |
| 4 | — | osqp | 3 | 3 | 100.0% | 100.0% | 1.8652e-17 | 0.0112603 | 0 | 0 |
| 4 | — | fractional-vqe | 3 | 9 | 100.0% | 33.3% | 0.00192282 | 0.211307 | 0 | 0 |
| 6 | 3 | exact-binary | 3 | 3 | 100.0% | 100.0% | 0 | 0.000366467 | 0 | 0 |
| 6 | 3 | highs | 3 | 3 | 100.0% | 100.0% | 0 | 0.012553 | 0 | 0 |
| 6 | 3 | swap | 3 | 9 | 100.0% | 100.0% | 0 | 0.00193695 | 0 | 0 |
| 6 | 3 | random-k | 3 | 9 | 100.0% | 100.0% | 0 | 0.00617201 | 0 | 0 |
| 6 | 3 | binary-vqe | 3 | 9 | 100.0% | 33.3% | 0.0249047 | 0.820233 | 0 | 0 |
| 6 | 3 | qaoa-x | 3 | 9 | 100.0% | 100.0% | 0 | 1.21391 | 0 | 0 |
| 6 | 3 | qaoa-xy | 3 | 9 | 100.0% | 66.7% | 0.0169091 | 1.37852 | 0 | 0 |
| 6 | — | exact-fractional | 3 | 3 | 100.0% | 100.0% | 0 | 0.00421969 | 0 | 0 |
| 6 | — | osqp | 3 | 3 | 100.0% | 100.0% | 1.05356e-17 | 0.00850309 | 0 | 0 |
| 6 | — | fractional-vqe | 3 | 9 | 100.0% | 11.1% | 0.00489347 | 0.280956 | 0 | 0 |
| 8 | 4 | exact-binary | 3 | 3 | 100.0% | 100.0% | 0 | 0.000722476 | 0 | 0 |
| 8 | 4 | highs | 3 | 3 | 100.0% | 100.0% | 0 | 0.0240882 | 0 | 0 |
| 8 | 4 | swap | 3 | 9 | 100.0% | 100.0% | 0 | 0.00244037 | 0 | 0 |
| 8 | 4 | random-k | 3 | 9 | 100.0% | 100.0% | 0 | 0.00587656 | 0 | 0 |
| 8 | 4 | binary-vqe | 3 | 9 | 100.0% | 0.0% | 0.0308508 | 1.38211 | 0 | 0 |
| 8 | 4 | qaoa-x | 3 | 9 | 100.0% | 100.0% | 0 | 2.07311 | 0 | 0 |
| 8 | 4 | qaoa-xy | 3 | 9 | 100.0% | 0.0% | 0.0354287 | 2.22395 | 0 | 0 |
| 8 | — | exact-fractional | 3 | 3 | 100.0% | 100.0% | 0 | 0.0194856 | 0 | 0 |
| 8 | — | osqp | 3 | 3 | 100.0% | 100.0% | -5.42719e-17 | 0.0105921 | 0 | 0 |
| 8 | — | fractional-vqe | 3 | 9 | 100.0% | 33.3% | 0.00453815 | 0.34082 | 0 | 0 |

See manifest.json for frozen data, settings, references, software versions and source hashes; trials.json is the resumable checkpoint. Sample feasibility and Top-K fallback use are in the trial records.
