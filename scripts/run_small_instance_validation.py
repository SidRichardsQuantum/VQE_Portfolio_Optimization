"""Run/checkpoint a configured small-instance simulator validation study."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from vqe_portfolio.research.infrastructure import provenance, write_csv, write_json
from vqe_portfolio.research.validation import (
    ValidationConfig,
    evaluate_trial,
    prepare_instances,
    summarise_trials,
    trial_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def export_results(output, trials, expected):
    summaries = summarise_trials(trials)
    for filename, rows in (("trials.csv", trials), ("summary.csv", summaries)):
        write_csv(output / filename, rows, json_fields=("candidate",))
    lines = [
        "# Small-instance validation pilot",
        "",
        f"Completed {len(trials)} of {expected} planned trials.",
        "",
        "Synthetic factor-model instances; noiseless default.qubit simulator; exact-expectation training and finite-shot binary sampling.",
        "",
        "This is a correctness/reliability pilot, not a fixed-budget study or a quantum-advantage test. Timings describe this simulator and machine only. Reference preparation is excluded from solver runtimes. Deterministic solvers run once per instance; stochastic methods use every configured solver seed.",
        "",
        "Target: feasible normalised objective gap <= the configured target_gap (default 0.001). Gap scale is fixed per instance, not a percentage investment loss. Binary and fractional objectives are separate tracks. Fractional references are numerical active-set solutions, not exact-arithmetic certificates.",
        "",
        "Rates and runtimes average instance means. Quality averages feasible-run means within each instance and excludes instances without a feasible return; inspect quality_instance_count and all trial failures. No confidence intervals or significance claims are made for this pilot.",
        "",
        "| Assets | K | Method | Instances | Trials | Feasible | Target hit | Mean gap (feasible) | Mean seconds | Errors | Reference contradictions |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        gap = row["mean_normalised_gap_feasible"]
        gap_text = "—" if gap is None else f"{gap:.6g}"
        lines.append(
            f"| {row['n_assets']} | {row['k'] if row['k'] is not None else '—'} | {row['method']} | {row['instance_count']} | {row['trial_count']} | {row['feasible_rate']:.1%} | {row['target_hit_rate']:.1%} | {gap_text} | {row['mean_runtime_seconds']:.6g} | {row['error_count']} | {row['reference_contradictions']} |"
        )
    lines += [
        "",
        "See manifest.json for frozen data, settings, references, software versions and source hashes; trials.json is the resumable checkpoint. Sample feasibility and Top-K fallback use are in the trial records.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/small_instance_validation.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/small_instance_validation"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only with matching configuration, source hashes and environment.",
    )
    args = parser.parse_args(argv)
    config = ValidationConfig(**json.loads(args.config.read_text()))
    # Fail visibly for absent optional dependencies before creating a study.
    if "scip" in config.methods:
        import pyscipopt  # noqa: F401
    if "osqp" in config.methods:
        import cvxpy  # noqa: F401
        import osqp  # noqa: F401
    if "highs" in config.methods:
        from scipy.optimize import milp  # noqa: F401
    output = args.output_dir
    identity = {
        "schema_version": 1,
        "config": config.to_dict(),
        "environment": provenance(ROOT, Path(__file__)),
    }
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if any(manifest[key] != value for key, value in identity.items()):
            raise ValueError(
                "Resume refused: configuration, source or environment changed. Use a new output directory."
            )
        trials = json.loads((output / "trials.json").read_text())
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                "Output directory is nonempty; use --resume or a new directory."
            )
        instances = prepare_instances(config)
        output.mkdir(parents=True, exist_ok=True)
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        manifest = {
            **identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": git.stdout.strip() if git.returncode == 0 else None,
            "execution": "serial, noiseless simulator, fixed optimisation steps; no end-to-end budget enforcement",
            "instances": instances,
        }
        write_json(output / "manifest.json", manifest)
        trials = []
        write_json(output / "trials.json", trials)
    plan = list(trial_plan(manifest["instances"], config))
    completed = {row["trial_id"] for row in trials}
    if len(completed) != len(trials) or completed - {entry[0] for entry in plan}:
        raise ValueError("Checkpoint contains duplicate or unplanned trials.")
    try:
        for key, instance, method, seed in plan:
            if key in completed:
                continue
            trial = evaluate_trial(key, instance, method, seed, config)
            trials.append(trial)
            write_json(output / "trials.json", trials)
            print(
                f"[{len(trials)}/{len(plan)}] n={trial['n_assets']} K={trial['k']} instance={trial['instance_seed']} {method} seed={seed}: feasible={trial['feasible']} target={trial['target_hit']} {trial['runtime_seconds']:.3f}s",
                flush=True,
            )
    finally:
        export_results(output, trials, len(plan))
    print(f"Results: {output / 'report.md'}")
    return int(any(row["error"] or row["reference_contradiction"] for row in trials))


if __name__ == "__main__":
    raise SystemExit(main())
