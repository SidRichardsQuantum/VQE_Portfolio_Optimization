# Small-instance validation pilot

Completed 128 of 128 planned trials.

Synthetic factor-model instances; noiseless default.qubit simulator; exact-expectation training and finite-shot binary sampling.

This is a correctness/reliability pilot, not a fixed-budget study or a quantum-advantage test. Timings describe this simulator and machine only. Reference preparation is excluded from solver runtimes. Deterministic solvers run once per instance; stochastic methods use every configured solver seed.

Target: feasible normalised objective gap <= the configured target_gap (default 0.001). Gap scale is fixed per instance, not a percentage investment loss. Binary and fractional objectives are separate tracks. Fractional references are numerical active-set solutions, not exact-arithmetic certificates.

Rates and runtimes average instance means. Quality averages feasible-run means within each instance and excludes instances without a feasible return; inspect quality_instance_count and all trial failures. No confidence intervals or significance claims are made for this pilot.

| Assets | K | Method | Instances | Trials | Feasible | Target hit | Mean gap (feasible) | Mean seconds | Errors | Reference contradictions |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 | exact-binary | 2 | 2 | 100.0% | 100.0% | 0 | 0.000139022 | 0 | 0 |
| 4 | 2 | top-return | 2 | 2 | 100.0% | 100.0% | 0 | 0.000297894 | 0 | 0 |
| 4 | 2 | highs | 2 | 2 | 100.0% | 100.0% | 0 | 0.0161179 | 0 | 0 |
| 4 | 2 | scip | 2 | 2 | 100.0% | 100.0% | 0 | 0.0112917 | 0 | 0 |
| 4 | 2 | qp-round | 2 | 2 | 100.0% | 100.0% | 0 | 0.0100281 | 0 | 0 |
| 4 | 2 | swap | 2 | 4 | 100.0% | 100.0% | 0 | 0.000424994 | 0 | 0 |
| 4 | 2 | anneal | 2 | 4 | 100.0% | 100.0% | 0 | 0.0156786 | 0 | 0 |
| 4 | 2 | tabu | 2 | 4 | 100.0% | 100.0% | 0 | 0.0245702 | 0 | 0 |
| 4 | 2 | memetic | 2 | 4 | 100.0% | 100.0% | 0 | 0.105964 | 0 | 0 |
| 4 | 2 | ils | 2 | 4 | 100.0% | 100.0% | 0 | 0.0439839 | 0 | 0 |
| 4 | 2 | grasp | 2 | 4 | 100.0% | 100.0% | 0 | 0.000600255 | 0 | 0 |
| 4 | 2 | vns | 2 | 4 | 100.0% | 100.0% | 0 | 0.0348184 | 0 | 0 |
| 4 | 2 | random-k | 2 | 4 | 100.0% | 100.0% | 0 | 0.00393046 | 0 | 0 |
| 4 | 2 | binary-vqe | 2 | 4 | 100.0% | 50.0% | 0.020028 | 0.362788 | 0 | 0 |
| 4 | 2 | qaoa-x | 2 | 4 | 100.0% | 100.0% | 0 | 0.376957 | 0 | 0 |
| 4 | 2 | qaoa-xy | 2 | 4 | 100.0% | 25.0% | 0.0394973 | 0.466868 | 0 | 0 |
| 4 | — | exact-fractional | 2 | 2 | 100.0% | 100.0% | 0 | 0.0021705 | 0 | 0 |
| 4 | — | osqp | 2 | 2 | 100.0% | 100.0% | 0 | 0.0117855 | 0 | 0 |
| 4 | — | pgd | 2 | 2 | 100.0% | 100.0% | 1.39026e-15 | 0.00570706 | 0 | 0 |
| 4 | — | variational-allocator | 2 | 4 | 100.0% | 25.0% | 0.0154888 | 0.120987 | 0 | 0 |
| 6 | 3 | exact-binary | 2 | 2 | 100.0% | 100.0% | 0 | 0.000279245 | 0 | 0 |
| 6 | 3 | top-return | 2 | 2 | 100.0% | 0.0% | 0.0471975 | 0.000218439 | 0 | 0 |
| 6 | 3 | highs | 2 | 2 | 100.0% | 100.0% | 0 | 0.0134833 | 0 | 0 |
| 6 | 3 | scip | 2 | 2 | 100.0% | 100.0% | 0 | 0.022475 | 0 | 0 |
| 6 | 3 | qp-round | 2 | 2 | 100.0% | 100.0% | 0 | 0.00923515 | 0 | 0 |
| 6 | 3 | swap | 2 | 4 | 100.0% | 100.0% | 0 | 0.000526555 | 0 | 0 |
| 6 | 3 | anneal | 2 | 4 | 100.0% | 100.0% | 0 | 0.0160781 | 0 | 0 |
| 6 | 3 | tabu | 2 | 4 | 100.0% | 100.0% | 0 | 0.0226501 | 0 | 0 |
| 6 | 3 | memetic | 2 | 4 | 100.0% | 100.0% | 0 | 0.116686 | 0 | 0 |
| 6 | 3 | ils | 2 | 4 | 100.0% | 100.0% | 0 | 0.0400989 | 0 | 0 |
| 6 | 3 | grasp | 2 | 4 | 100.0% | 100.0% | 0 | 0.000776095 | 0 | 0 |
| 6 | 3 | vns | 2 | 4 | 100.0% | 100.0% | 0 | 0.0425595 | 0 | 0 |
| 6 | 3 | random-k | 2 | 4 | 100.0% | 100.0% | 0 | 0.00384111 | 0 | 0 |
| 6 | 3 | binary-vqe | 2 | 4 | 100.0% | 25.0% | 0.00798853 | 0.565084 | 0 | 0 |
| 6 | 3 | qaoa-x | 2 | 4 | 100.0% | 100.0% | 0 | 0.809285 | 0 | 0 |
| 6 | 3 | qaoa-xy | 2 | 4 | 100.0% | 0.0% | 0.0425636 | 1.01672 | 0 | 0 |
| 6 | — | exact-fractional | 2 | 2 | 100.0% | 100.0% | 0 | 0.00532639 | 0 | 0 |
| 6 | — | osqp | 2 | 2 | 100.0% | 100.0% | -3.41531e-17 | 0.00893235 | 0 | 0 |
| 6 | — | pgd | 2 | 2 | 100.0% | 100.0% | 3.17732e-15 | 0.0038833 | 0 | 0 |
| 6 | — | variational-allocator | 2 | 4 | 100.0% | 25.0% | 0.00804691 | 0.165899 | 0 | 0 |

See manifest.json for frozen data, settings, references, software versions and source hashes; trials.json is the resumable checkpoint. Sample feasibility and Top-K fallback use are in the trial records.
