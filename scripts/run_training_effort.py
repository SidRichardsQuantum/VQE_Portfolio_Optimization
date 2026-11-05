"""Checkpoint one continuous training run at several optimisation efforts."""

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from vqe_portfolio.research.infrastructure import provenance, write_csv, write_json
from vqe_portfolio.research.sampling_allowance import (
    sampling_rows,
    validate_draw_counts,
)
from vqe_portfolio.research.training_diagnostics import (
    DIAGNOSTIC_METHODS,
    validate_checkpoints,
)
from vqe_portfolio.research.training_effort import measure_trajectory, summarise_effort
from vqe_portfolio.research.validation import ValidationConfig, trial_plan

ROOT = Path(__file__).resolve().parents[1]


COLORS = ("#2563eb", "#b45309", "#15803d")


def plot_results(output, trajectories, summaries, steps, draws):
    import matplotlib.pyplot as plt

    sizes = sorted({(r["n_assets"], r["k"]) for r in summaries})
    chosen_draws = max(draws)
    metrics = (
        ("p_target", "Target probability (%)", 100),
        ("p_feasible", "Feasibility (%)", 100),
        (f"success_{chosen_draws}", f"Success in {chosen_draws} draws (%)", 100),
        ("training_runtime_seconds", "Cumulative training seconds", 1),
    )
    figure, axes = plt.subplots(
        4, len(sizes), figsize=(max(9, 4.1 * len(sizes)), 10.7), squeeze=False
    )
    for j, (n, k) in enumerate(sizes):
        for i, (field, label, scale) in enumerate(metrics):
            axis = axes[i, j]
            for method, color in zip(DIAGNOSTIC_METHODS, COLORS):
                rows = sorted(
                    (
                        r
                        for r in summaries
                        if (r["n_assets"], r["k"], r["method"]) == (n, k, method)
                        and r["successful_trajectories"]
                    ),
                    key=lambda r: r["step"],
                )
                axis.plot(
                    [r["step"] for r in rows],
                    [scale * r[f"mean_{field}"] for r in rows],
                    color=color,
                    marker="o",
                    markersize=4,
                    label=method,
                )
            axis.set_xticks(steps)
            axis.set_ylim(bottom=0)
            if scale == 100:
                axis.set_ylim(0, 103)
            axis.grid(alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
            if j == 0:
                axis.set_ylabel(label)
            if i == 0:
                axis.set_title(f"{n} securities, K={k}")
            if i == 3:
                axis.set_xlabel("Optimiser steps")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Training effort along the same optimisation trajectory", y=0.99)
    figure.subplots_adjust(
        top=0.94, bottom=0.07, left=0.08, right=0.98, hspace=0.3, wspace=0.24
    )
    figure.savefig(output / "training_effort.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        3, len(sizes), figsize=(max(9, 4.1 * len(sizes)), 8.5), squeeze=False
    )
    for i, method in enumerate(DIAGNOSTIC_METHODS):
        for j, (n, k) in enumerate(sizes):
            axis = axes[i, j]
            for trial in trajectories:
                if (trial["n_assets"], trial["k"], trial["method"]) != (
                    n,
                    k,
                    method,
                ) or trial["error"]:
                    continue
                axis.plot(
                    [s["step"] for s in trial["checkpoints"]],
                    [
                        100 * s["p_target"] if s["p_target"] >= 1e-12 else 0
                        for s in trial["checkpoints"]
                    ],
                    color=COLORS[i],
                    alpha=0.3,
                    marker="o",
                    markersize=3,
                    linewidth=1,
                )
            rows = sorted(
                (
                    r
                    for r in summaries
                    if (r["n_assets"], r["k"], r["method"]) == (n, k, method)
                    and r["successful_trajectories"]
                ),
                key=lambda r: r["step"],
            )
            axis.plot(
                [r["step"] for r in rows],
                [
                    100 * r["mean_p_target"] if r["mean_p_target"] >= 1e-12 else 0
                    for r in rows
                ],
                color="#111827",
                marker="D",
                linewidth=2.2,
                markersize=4,
            )
            axis.set_title(f"{method}: n={n}, K={k}")
            axis.set_xticks(steps)
            axis.set_ylim(bottom=0)
            if not rows or max(r["mean_p_target"] for r in rows) < 1e-12:
                axis.set_ylim(0, 1)
            axis.grid(alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
            if j == 0:
                axis.set_ylabel("Target probability (%)")
            if i == 2:
                axis.set_xlabel("Optimiser steps")
    figure.suptitle(
        "Paired runs: target-quality probability versus training effort", y=0.99
    )
    figure.text(
        0.5,
        0.94,
        "Thin curves: individual runs. Black: instance-weighted mean. Each panel has its own probability scale.",
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(
        top=0.87, bottom=0.08, left=0.08, right=0.98, hspace=0.45, wspace=0.28
    )
    figure.savefig(output / "training_pairs.png", dpi=180)
    plt.close(figure)


def export_results(output, trajectories, steps, draws, expected):
    rows, summaries, paired = summarise_effort(trajectories, steps, draws)
    for filename, records in (
        ("checkpoints.csv", rows),
        ("summary.csv", summaries),
        ("paired_summary.csv", paired),
    ):
        write_csv(output / filename, records)

    def percent(p):
        return "—" if p is None else "<1e-10%" if 0 < p < 1e-12 else f"{100*p:.4g}%"

    lines = [
        "# Training-effort development study",
        "",
        f"Completed {len(trajectories)} of {expected} planned trajectories at checkpoints {', '.join(map(str, steps))}.",
        "",
        "Each trajectory initialises once and runs Adam continuously to the final step, preserving its moment estimates. Parameters and cumulative optimisation times are copied at the requested checkpoints. Exact noiseless distributions are evaluated only after optimisation has finished. No samples, Top-K repairs, best-iterate selection or held-out tuning are used.",
        "",
        "The frozen problems, initialisation seeds, depths, penalties, learning rates and quality target are inherited from the prior diagnostic. When available, the initial and original training-step parameters/probabilities must match that evidence; a mismatch is recorded as a trajectory error.",
        "",
        "| Securities | K | Method | Step | Target probability | Feasibility | "
        + " | ".join(f"Success: {m} draws" for m in draws)
        + " | Training seconds | Valid / all |",
        "| ---: | ---: | --- | ---: | ---: | ---: | "
        + " | ".join("---:" for m in draws)
        + " | ---: | ---: |",
    ]
    for r in summaries:
        seconds = r["mean_training_runtime_seconds"]
        runtime = "—" if seconds is None else f"{seconds:.5g}"
        lines.append(
            f"| {r['n_assets']} | {r['k']} | {r['method']} | {r['step']} | {percent(r['mean_p_target'])} | {percent(r['mean_p_feasible'])} | "
            + " | ".join(percent(r[f"mean_success_{m}"]) for m in draws)
            + f" | {runtime} | {r['successful_trajectories']} / {r['trajectory_count']} |"
        )
    lines += [
        "",
        "Means give seeds equal weight within each instance, then give instances equal weight. Success probabilities use 1-(1-p)^m per run before averaging, counting all raw draws including infeasible ones. The target is a feasible portfolio within the frozen normalised objective-gap tolerance (0.001 in the pilot). Paired changes for every checkpoint interval are in paired_summary.csv; improved/worsened counts use a numerical 1e-12 probability deadband, not a significance test.",
        "",
        "Training time is cumulative wall time within the Adam loop, including callback/logging overhead and excluding problem encoding, device setup, diagnostic probability evaluation, sampling and portfolio postprocessing. Step zero has zero optimiser time. These descriptive simulator timings are not a common end-to-end budget or QPU measurement.",
        "",
        "Failed trajectories remain in counts and have blank checkpoint metrics; probability means use complete successful trajectories only, with coverage explicit in quality_instance_count and successful_trajectories. Resume skips completed trajectories, including errors. Interrupted trajectories restart from their original seed; parameter snapshots alone are not treated as resumable Adam state.",
        "",
        "This is development sensitivity analysis on the existing small instance set. It provides no confidence intervals, held-out evaluation, scaling law or quantum-advantage claim. More training can improve the penalty energy or feasibility without improving target probability. Connecting plot lines link evaluated checkpoints rather than showing every intermediate iterate. Plot values below 1e-12 probability are displayed as zero; full precision is retained in the data.",
        "",
        "![Training effort, reliability and cumulative time](training_effort.png)",
        "",
        "![Paired individual training trajectories](training_pairs.png)",
        "",
    ]
    if summaries:
        plot_results(output, trajectories, summaries, steps, draws)
    (output / "report.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-dir", type=Path, default=ROOT / "results/training_diagnostic"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/training_effort"
    )
    parser.add_argument("--steps", default="0,25,100")
    parser.add_argument("--draws", default="16,64,512")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    requested = tuple(int(s) for s in args.steps.split(","))
    steps = validate_checkpoints(requested, max(requested))
    draws = validate_draw_counts(tuple(int(s) for s in args.draws.split(",")))
    inputs = {
        name: (args.diagnostic_dir / name).read_bytes()
        for name in ("manifest.json", "pairs.json")
    }
    source = json.loads(inputs["manifest.json"])
    if (
        source.get("schema_version") != 1
        or source.get("study") != "initial_vs_trained_binary_distributions"
    ):
        raise ValueError("Require a version 1 training diagnostic manifest.")
    source_records = json.loads(inputs["pairs.json"])
    sampling_rows(source, source_records, (1,))  # verify the complete frozen input
    source_records = {r["trial_id"]: r for r in source_records}
    config = replace(ValidationConfig(**source["config"]), steps=max(steps))
    if set(config.methods) - set(DIAGNOSTIC_METHODS):
        raise ValueError("Source must contain only the binary diagnostic methods.")
    environment = provenance(ROOT, Path(__file__), extra_packages=("matplotlib",))
    identity = {
        "schema_version": 1,
        "study": "continuous_training_effort",
        "config": config.to_dict(),
        "checkpoint_steps": list(steps),
        "draw_counts": list(draws),
        "source_inputs_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()
        },
        "environment": environment,
    }
    output = args.output_dir
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if any(manifest.get(k) != value for k, value in identity.items()):
            raise ValueError(
                "Resume refused: source, configuration or environment changed."
            )
        trajectories = json.loads((output / "trajectories.json").read_text())
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                "Output directory is nonempty; use --resume or a new directory."
            )
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            **identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "instances": source["instances"],
            "sampling_performed": False,
        }
        write_json(output / "manifest.json", manifest)
        trajectories = []
        write_json(output / "trajectories.json", trajectories)
    plan = list(trial_plan(manifest["instances"], config))
    completed = {r["trial_id"] for r in trajectories}
    if len(completed) != len(trajectories) or completed - {entry[0] for entry in plan}:
        raise ValueError("Checkpoint contains duplicate or unplanned trajectories.")
    try:
        for key, instance, method, seed in plan:
            if key in completed:
                continue
            row = measure_trajectory(
                key, instance, method, seed, config, steps, draws, source_records[key]
            )
            trajectories.append(row)
            write_json(output / "trajectories.json", trajectories)
            status = (
                row["error"]
                or f"P(target): {row['checkpoints'][0]['p_target']:.5g} -> {row['checkpoints'][-1]['p_target']:.5g}"
            )
            print(
                f"[{len(trajectories)}/{len(plan)}] n={row['n_assets']} {method} instance={row['instance_seed']} seed={seed}: {status}",
                flush=True,
            )
    finally:
        export_results(output, trajectories, steps, draws, len(plan))
    print(f"Report: {output / 'report.md'}")
    return int(any(r["error"] for r in trajectories))


if __name__ == "__main__":
    raise SystemExit(main())
