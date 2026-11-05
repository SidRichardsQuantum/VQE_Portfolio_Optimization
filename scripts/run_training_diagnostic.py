"""Measure initial/trained target probabilities on frozen binary pilot problems."""

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from vqe_portfolio.training_diagnostics import (
    DIAGNOSTIC_METHODS,
    measure_pair,
    summarise_pairs,
)
from vqe_portfolio.validation import ValidationConfig, trial_plan

if __package__:
    from .run_small_instance_validation import ROOT, provenance, write_json
else:
    from run_small_instance_validation import ROOT, provenance, write_json


def plot_results(output, summaries):
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, method in zip(axes, DIAGNOSTIC_METHODS):
        rows = [r for r in summaries if r["method"] == method and r["successful_pairs"]]
        positions = np.arange(len(rows))
        for offset, field, label, color in (
            (-0.18, "before", "Initial", "#64748b"),
            (0.18, "after", "Trained", "#2563eb"),
        ):
            values = [100 * r[f"mean_{field}_p_target"] for r in rows]
            bars = axis.bar(
                positions + offset, values, width=0.36, label=label, color=color
            )
            axis.bar_label(
                bars,
                labels=[
                    "≈0%" if value < 1e-10 else f"{value:.2g}%" for value in values
                ],
                fontsize=8,
                padding=3 if offset < 0 else 20,
            )
        axis.set_xticks(positions, [r["n_assets"] for r in rows])
        axis.set_xlabel("Securities")
        axis.set_title(method)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(y=0.2)
    axes[0].set_ylabel("Target-quality feasible probability per draw (%)")
    axes[0].legend(frameon=False)
    figure.suptitle(
        "Before and after training — exact noiseless simulator distributions"
    )
    figure.tight_layout()
    figure.savefig(output / "training_probability.png", dpi=180)
    plt.close(figure)


def export_results(output, trials, expected):
    summaries = summarise_pairs(trials)
    omitted = {
        "initial_params",
        "trained_params",
        "before_probabilities",
        "after_probabilities",
        "algorithm_config",
        "target_basis_indices",
    }
    for filename, rows in (
        (
            "pairs.csv",
            [{k: v for k, v in r.items() if k not in omitted} for r in trials],
        ),
        ("summary.csv", summaries),
    ):
        if rows:
            with (output / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=list(dict.fromkeys(k for row in rows for k in row)),
                )
                writer.writeheader()
                writer.writerows(rows)

    def percent(value):
        return "—" if value is None else f"{100 * value:.4g}%"

    lines = [
        "# Before-and-after training diagnostic",
        "",
        f"Completed {len(trials)} of {expected} planned initial/trained pairs.",
        "",
        "Each pair uses the same frozen binary problem, circuit, initial parameters and seed. Training settings are inherited from the pilot (25 Adam steps in the default experiment). Initial means random parameters before any optimiser update, with the full circuit applied. Trained means the final parameters after all updates; no best-iterate selection is used.",
        "",
        "The primary measure is the exact single-draw probability of a feasible portfolio with normalised original-objective gap <= the configured target_gap (0.001 in the pilot). It is not a 512-sample hit rate or a postprocessed portfolio score. No samples or Top-K repairs are used. Reference scores are used only by the evaluator, never by training.",
        "",
        "| Securities | K | Method | Successful pairs / all | Initial P(target) | Trained P(target) | Change (percentage points) | Improved pairs | Worsened pairs | Initial P(feasible) | Trained P(feasible) |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        change = row["mean_delta_p_target"]
        change_text = "—" if change is None else f"{100 * change:+.4g}"
        lines.append(
            f"| {row['n_assets']} | {row['k']} | {row['method']} | {row['successful_pairs']} / {row['pair_count']} | {percent(row['mean_before_p_target'])} | {percent(row['mean_after_p_target'])} | {change_text} | {row['improved_pairs']} | {row['worsened_pairs']} | {percent(row['mean_before_p_feasible'])} | {percent(row['mean_after_p_feasible'])} |"
        )
    lines += [
        "",
        "Metrics average seeds within each instance, then give instances equal weight. Paired changes are computed within runs before aggregation. Improved/worsened counts use a numerical deadband of 1e-12 probability, not a statistical significance threshold. Failed pairs remain in the counts and are excluded from probability means; inspect error_count and quality_instance_count in summary.csv.",
        "",
        "pairs.csv also reports P(target | feasible), expected penalised energy before/after, and uniform-bitstring/uniform-feasible single-draw reference probabilities. Conditional probabilities average per-run ratios, rather than ratios of pooled means. A circuit can improve feasibility or its penalised training energy while making target portfolios less likely.",
        "",
        "The checkpoint pairs.json retains initial/final parameters, full probability vectors in computational-basis order (wire 0 is the most significant bit), target basis indices, and actual solver configurations. The configured shots_sample value is inherited for traceability but is not used by this diagnostic. Sampling and depth/budget sweeps are outside this experiment. Fractional VQE returns an allocation rather than a sampled binary portfolio and is outside this diagnostic.",
        "",
        "These are exact evaluations of a noiseless simulator, not QPU measurements or evidence of computational advantage. Only three independent instances per size are used in the pilot, with three initialisation seeds each; no confidence intervals or scaling claims are made.",
        "",
    ]
    if any(row["successful_pairs"] for row in summaries):
        plot_results(output, summaries)
        lines += [
            "![Exact target probabilities before and after training](training_probability.png)",
            "Plot labels below 1e-10% are shown as approximately zero; full precision is retained in the tables and checkpoint.",
            "",
        ]
    (output / "report.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=ROOT / "results/small_instance_validation/manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/training_diagnostic"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    pilot_bytes = args.pilot_manifest.read_bytes()
    pilot = json.loads(pilot_bytes)
    config = replace(ValidationConfig(**pilot["config"]), methods=DIAGNOSTIC_METHODS)
    instances = [
        i
        for i in pilot["instances"]
        if i["snapshot"]["objective_family"] == "binary_qubo"
    ]
    if not instances or len({i["problem_id"] for i in instances}) != len(instances):
        raise ValueError("Require distinct frozen binary pilot instances.")
    environment = provenance()
    environment["source_sha256"][str(Path(__file__).relative_to(ROOT))] = (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    identity = {
        "schema_version": 1,
        "study": "initial_vs_trained_binary_distributions",
        "pilot_manifest_sha256": hashlib.sha256(pilot_bytes).hexdigest(),
        "config": config.to_dict(),
        "environment": environment,
    }
    output = args.output_dir
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if any(manifest.get(k) != value for k, value in identity.items()):
            raise ValueError(
                "Resume refused: pilot, configuration, source or environment changed."
            )
        trials = json.loads((output / "pairs.json").read_text())
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                "Output directory is nonempty; use --resume or a new directory."
            )
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            **identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "sampling_performed": False,
            "instances": instances,
        }
        write_json(output / "manifest.json", manifest)
        trials = []
        write_json(output / "pairs.json", trials)
    plan = list(trial_plan(manifest["instances"], config))
    completed = {row["trial_id"] for row in trials}
    if len(completed) != len(trials) or completed - {entry[0] for entry in plan}:
        raise ValueError("Checkpoint contains duplicate or unplanned pairs.")
    try:
        for key, instance, method, seed in plan:
            if key in completed:
                continue
            row = measure_pair(key, instance, method, seed, config)
            trials.append(row)
            write_json(output / "pairs.json", trials)
            print(
                f"[{len(trials)}/{len(plan)}] n={row['n_assets']} {method} instance={row['instance_seed']} seed={seed}: {'ERROR ' + row['error'] if row['error'] else 'delta P(target)=' + format(row['delta_p_target'], '+.6g')}",
                flush=True,
            )
    finally:
        export_results(output, trials, len(plan))
    print(f"Report: {output / 'report.md'}")
    return int(any(row["error"] for row in trials))


if __name__ == "__main__":
    raise SystemExit(main())
