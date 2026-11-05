"""Investigate XY reachability and VQE depth/penalty failures on frozen inputs."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from vqe_portfolio.failure_diagnostics import (
    measure_variant,
    static_diagnostics,
    variant_plan,
)
from vqe_portfolio.training_effort import summarise_effort
from vqe_portfolio.validation import ValidationConfig, trial_plan

if __package__:
    from .run_small_instance_validation import ROOT, provenance, write_json
else:
    from run_small_instance_validation import ROOT, provenance, write_json


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def plot_results(
    output, analysis, summaries, steps, xy_depths, vqe_depths, penalties, draws
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    info = next(iter(analysis["problems"].values()))
    feasible = info["feasible_bitstrings"]
    rows = [[int(b in info[f"xy_support_{p}"]) for b in feasible] for p in xy_depths]
    labels = [f"p={p}: {'exact' if p == 1 else 'path bound'}" for p in xy_depths]
    for problem in analysis["problems"].values():
        rows.append([int(b in problem["target_bitstrings"]) for b in feasible])
        labels.append(f"Targets: instance {problem['instance_seed']}")
    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.imshow(
        rows, aspect="auto", cmap=ListedColormap(["#ffffff", "#2563eb"]), vmin=0, vmax=1
    )
    selected = set(info.get("xy_support_1", []))
    for problem in analysis["problems"].values():
        selected.update(problem["target_bitstrings"])
    ticks = [i for i, b in enumerate(feasible) if b in selected]
    axis.set_xticks(ticks, [feasible[i] for i in ticks], rotation=90, fontsize=8)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Feasible bitstrings in lexicographic order (selected labels)")
    axis.set_title(
        f"XY QAOA support and target portfolios: n={info['n_assets']}, K={info['k']}"
    )
    figure.text(
        0.5,
        0.02,
        "Blue: membership. Higher-depth path bounds ignore interference; inclusion alone does not prove reachability.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output / "xy_support.png", dpi=180)
    plt.close(figure)

    lookup = {
        (r["method"], r["step"]): r for r in summaries if r["successful_trajectories"]
    }
    final_step = max(steps)
    selected_draws = max(draws)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.3))
    for i, field in enumerate(("p_target", f"success_{selected_draws}")):
        axis = axes[i, 0]
        for step, color in zip(steps, ("#64748b", "#2563eb", "#b45309", "#15803d")):
            values = [
                (
                    100 * lookup[(f"xy-p{p}", step)][f"mean_{field}"]
                    if (f"xy-p{p}", step) in lookup
                    else np.nan
                )
                for p in xy_depths
            ]
            axis.plot(xy_depths, values, marker="o", label=f"{step} steps", color=color)
        axis.set_xticks(xy_depths)
        axis.set_ylim(0, 103)
        axis.set_xlabel("XY QAOA depth")
        axis.set_ylabel(
            "Target probability (%)"
            if i == 0
            else f"Success in {selected_draws} draws (%)"
        )
        axis.grid(alpha=0.2)
        if i == 0:
            axis.set_title("XY depth comparison")
            axis.legend(frameon=False, fontsize=9)
        values = [
            [
                (
                    100 * lookup[(f"vqe-d{d}-a{a:g}", final_step)][f"mean_{field}"]
                    if (f"vqe-d{d}-a{a:g}", final_step) in lookup
                    else np.nan
                )
                for a in penalties
            ]
            for d in vqe_depths
        ]
        axis = axes[i, 1]
        graphic = axis.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
        axis.set_xticks(range(len(penalties)), [f"{a:g}" for a in penalties])
        axis.set_yticks(range(len(vqe_depths)), vqe_depths)
        axis.set_xlabel("VQE penalty alpha")
        axis.set_ylabel("VQE depth")
        for y, row in enumerate(values):
            for x, value in enumerate(row):
                label = (
                    "—"
                    if np.isnan(value)
                    else "≈0%" if value < 1e-8 else f"{value:.3g}%"
                )
                axis.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    color="white" if value > 55 else "black",
                )
        axis.set_title(
            f"VQE at {final_step} steps: {'target' if i == 0 else 'sampling success'}"
        )
        figure.colorbar(graphic, ax=axis, label="%")
    figure.suptitle(f"{info['n_assets']}-security development sensitivity study")
    figure.tight_layout()
    figure.savefig(output / "failure_sensitivity.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(vqe_depths),
        len(penalties),
        figsize=(max(9, 4 * len(penalties)), 3.1 * len(vqe_depths)),
        squeeze=False,
    )
    for i, depth in enumerate(vqe_depths):
        for j, alpha in enumerate(penalties):
            axis = axes[i, j]
            variant = f"vqe-d{depth}-a{alpha:g}"
            for field, label, color in (
                ("p_feasible", "Feasible", "#15803d"),
                ("p_target", "Target per draw", "#2563eb"),
                (
                    f"success_{selected_draws}",
                    f"Success in {selected_draws} draws",
                    "#b45309",
                ),
            ):
                axis.plot(
                    steps,
                    [
                        (
                            100 * lookup[(variant, step)][f"mean_{field}"]
                            if (variant, step) in lookup
                            else np.nan
                        )
                        for step in steps
                    ],
                    label=label,
                    color=color,
                    marker="o",
                )
            axis.set_title(f"VQE depth {depth}, alpha={alpha:g}")
            axis.set_xticks(steps)
            axis.set_ylim(0, 103)
            axis.grid(alpha=0.2)
            if i == len(vqe_depths) - 1:
                axis.set_xlabel("Optimiser steps")
            if j == 0:
                axis.set_ylabel("Probability (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.subplots_adjust(bottom=0.13, top=0.93, hspace=0.35, wspace=0.2)
    figure.savefig(output / "vqe_convergence.png", dpi=180)
    plt.close(figure)


def export_results(output, trials, analysis, steps, draws, settings, expected):
    renamed = [{**r, "method": r["variant"]} for r in trials]
    checkpoints, summaries, paired = summarise_effort(renamed, steps, draws)
    for filename, rows in (
        ("support.csv", analysis["support_rows"]),
        ("beta_grid.csv", analysis["beta_rows"]),
        ("parameter_sensitivity.csv", analysis["parameter_rows"]),
        ("penalty_guarantees.csv", analysis["penalty_rows"]),
        ("checkpoints.csv", checkpoints),
        ("summary.csv", summaries),
        ("paired_summary.csv", paired),
    ):
        write_csv(output / filename, rows)
    modes = []
    for trial in trials:
        if not trial["error"] and "final_vqe_diagnostic" in trial:
            diagnostic = trial["final_vqe_diagnostic"]
            modes.append(
                {
                    "trial_id": trial["trial_id"],
                    "instance_seed": trial["instance_seed"],
                    "variant": trial["variant"],
                    "seed": trial["seed"],
                    **{
                        k: v
                        for k, v in diagnostic.items()
                        if k != "product_swap_barrier"
                    },
                    "product_swap_barrier": (
                        diagnostic["product_swap_barrier"] or {}
                    ).get("barrier"),
                }
            )
    write_csv(output / "vqe_modes.csv", modes)

    def percent(p):
        return "—" if p is None else "≈0%" if p < 1e-12 else f"{100*p:.4g}%"

    lines = [
        "# Eight-security failure diagnostics",
        "",
        f"Completed {len(trials)} of {expected} variant trajectories; {sum(r['reused_baseline'] for r in trials)} reuse frozen baseline trajectories, verified against the current production circuit.",
        "",
        "## XY support and parameter roles",
        "",
        "At depth 1 the diagonal cost layer acts on a single computational-basis initial portfolio and contributes only a global phase. Gamma cannot change the output probabilities. The exact mixer amplitudes are saved as integer Laurent polynomials in z=exp(2i beta), divided by 2 to the number of ordered exchange edges. Each exchange has stay amplitude (z+z^-1)/2 and swap amplitude (z^-1-z)/2. Exact integer cancellation identifies the support over all beta, rather than inferring impossibility from a numerical grid.",
        "",
        "For p>1 the gate-path support is an upper bound only: diagonal cost layers add no support, and each unequal exchange can stay or swap. Exclusion certifies unreachability; inclusion does not prove reachability because tied parameters and interference may reduce support. The first gamma remains globally redundant at all depths, while later gammas can affect the distribution. parameter_sensitivity.csv records finite +0.1 perturbation probes at fixed generic parameters generated with seed 314+depth; these probes supplement the analytic argument.",
        "",
        "| Instance | Depth | Support kind | Support count | Targets covered / all |",
        "| ---: | ---: | --- | ---: | ---: |",
    ]
    for r in analysis["support_rows"]:
        lines.append(
            f"| {r['instance_seed']} | {r['depth']} | {r['support_kind']} | {r['support_count']} | {r['covered_targets']} / {r['target_count']} |"
        )
    lines += [
        "",
        "## Penalty adequacy and VQE expressivity",
        "",
        "The strict sufficient penalty is alpha > max over infeasible x of (f_star-f(x))/(sum(x)-K)^2, clipped below at zero. Exhaustive floating-point enumeration supplies this numerical threshold and the penalised ground-state feasibility check. A sufficient penalty guarantees the global penalised optimum is feasible; it does not guarantee Adam finds it. VQE can represent every basis portfolio at either tested depth: set the first RY layer to pi*x and the remaining rotations to zero; CZ gates change phases only. Target-state constructions and their measured probabilities are saved separately from training and are never supplied to the optimiser.",
        "",
        "| Instance | Alpha | Critical alpha | Strict sufficient | All numerical minimisers feasible |",
        "| ---: | ---: | ---: | --- | --- |",
    ]
    for r in analysis["penalty_rows"]:
        lines.append(
            f"| {r['instance_seed']} | {r['alpha']:g} | {r['critical_alpha']:.6g} | {r['strict_sufficient']} | {r['all_numerical_minimisers_feasible']} |"
        )
    lines += [
        "",
        "## Fixed-effort depth and penalty comparisons",
        "",
        "All variants use the frozen original objectives, seeds, learning rates and 100-step effort (or the inherited source effort for a custom input). Changing depth changes parameter dimension and the production random initialisation; the same seed rule does not imply identical initial states across depths. Baselines reuse their historical distributions/timers; new variants run serially, so mixed historical runtimes are descriptive and not a speed comparison. No best-iterate selection, sampling, Top-K repair or held-out tuning occurs.",
        "",
        "| Variant | Step | Target per draw | Feasible | "
        + " | ".join(f"Success: {m} draws" for m in draws)
        + " | Valid / all |",
        "| --- | ---: | ---: | ---: | "
        + " | ".join("---:" for _ in draws)
        + " | ---: |",
    ]
    for r in summaries:
        if r["step"] == max(steps):
            lines.append(
                f"| {r['method']} | {r['step']} | {percent(r['mean_p_target'])} | {percent(r['mean_p_feasible'])} | "
                + " | ".join(percent(r[f"mean_success_{m}"]) for m in draws)
                + f" | {r['successful_trajectories']} / {r['trajectory_count']} |"
            )
    lines += [
        "",
        "## Convergence diagnostics and interpretation",
        "",
        "vqe_modes.csv records the final modal portfolio, its feasibility/objective gap, exact best one-for-one swap improvement, and final analytic energy-gradient norm. Gradient norms are descriptive, not local-optimum certificates. When an improving swap exists, the stored product-state path rotates the removed/added bits together: with u=sin^2(pi*t/2), its energy is f0+linear*u+quadratic*u^2. The computed maximum rise illustrates that particular path; it is not a lower bound on all entangled circuit paths or proof of a trapping mechanism.",
        "",
        "Means average seeds within each instance and then give instances equal weight. Success probabilities use each run's exact target mass before the nonlinear iid draw calculation. Failures remain in all counts and have blank metrics. Method labels in summary/checkpoint tables are variant identifiers. raw trajectories and static_analysis.json retain parameters, distributions, exact support coefficients and construction evidence. The beta grid is exploratory; its maximum is not a certified global maximum unless exact support excludes every target.",
        "",
        "These are small development-instance diagnostics, not held-out evidence of tuned performance, confidence intervals, scaling laws or quantum advantage. Penalty strengths are algorithm settings; returned quality always uses the same penalty-free feasible-portfolio objective. Approximately zero means below 1e-12 probability; percentages near certainty may round to 100%. CSVs retain full precision.",
        "",
        "![Support and target bitstrings](xy_support.png)",
        "",
        "![Depth and penalty sensitivity](failure_sensitivity.png)",
        "",
        "![VQE convergence](vqe_convergence.png)",
        "",
    ]
    if summaries:
        plot_results(
            output,
            analysis,
            summaries,
            steps,
            settings["xy_depths"],
            settings["vqe_depths"],
            settings["penalties"],
            draws,
        )
    (output / "report.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--effort-dir", type=Path, default=ROOT / "results/training_effort"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/failure_diagnostics"
    )
    parser.add_argument("--assets", type=int, default=8)
    parser.add_argument("--xy-depths", default="1,2,3")
    parser.add_argument("--vqe-depths", default="2,3")
    parser.add_argument("--penalties", default="0.05,0.5,5")
    parser.add_argument("--beta-points", type=int, default=257)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    settings = {
        "assets": args.assets,
        "xy_depths": [int(x) for x in args.xy_depths.split(",")],
        "vqe_depths": [int(x) for x in args.vqe_depths.split(",")],
        "penalties": [float(x) for x in args.penalties.split(",")],
        "beta_points": args.beta_points,
    }
    for name in ("xy_depths", "vqe_depths", "penalties"):
        values = settings[name]
        if (
            not values
            or len(set(values)) != len(values)
            or any(not np.isfinite(x) or x <= 0 for x in values)
        ):
            raise ValueError("Use distinct positive depths/penalties.")
    if args.beta_points < 3:
        raise ValueError("beta-points must be at least three.")
    inputs = {
        name: (args.effort_dir / name).read_bytes()
        for name in ("manifest.json", "trajectories.json")
    }
    source = json.loads(inputs["manifest.json"])
    if (
        source.get("study") != "continuous_training_effort"
        or source.get("schema_version") != 1
    ):
        raise ValueError("Require a version 1 continuous training-effort study.")
    base = ValidationConfig(**source["config"])
    source_records = json.loads(inputs["trajectories.json"])
    expected = {key for key, _, _, _ in trial_plan(source["instances"], base)}
    if {r["trial_id"] for r in source_records} != expected or len(
        source_records
    ) != len(expected):
        raise ValueError("Require a complete source trajectory checkpoint.")
    baseline_lookup = {
        (r["problem_id"], r["method"], r["seed"]): r for r in source_records
    }
    instances = [
        i for i in source["instances"] if len(i["snapshot"]["mu"]) == args.assets
    ]
    if not instances:
        raise ValueError("No frozen instances at the requested size.")
    steps, draws = source["checkpoint_steps"], source["draw_counts"]
    environment = provenance()
    environment["packages"]["matplotlib"] = version("matplotlib")
    environment["source_sha256"][str(Path(__file__).relative_to(ROOT))] = (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    identity = {
        "schema_version": 1,
        "study": "binary_failure_diagnostics",
        "config": base.to_dict(),
        "settings": settings,
        "checkpoint_steps": steps,
        "draw_counts": draws,
        "source_inputs_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()
        },
        "environment": environment,
    }
    output = args.output_dir
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError("Resume refused: source, settings or environment changed.")
        trials = json.loads((output / "trajectories.json").read_text())
        analysis = json.loads((output / "static_analysis.json").read_text())
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                "Output directory is nonempty; use --resume or a new directory."
            )
        analysis = static_diagnostics(
            instances,
            settings["xy_depths"],
            settings["vqe_depths"],
            settings["penalties"],
            base.target_gap,
            args.beta_points,
        )
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "manifest.json",
            {
                **identity,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "instances": instances,
            },
        )
        write_json(output / "static_analysis.json", analysis)
        trials = []
        write_json(output / "trajectories.json", trials)
    plan = list(
        variant_plan(
            base,
            instances,
            settings["xy_depths"],
            settings["vqe_depths"],
            settings["penalties"],
        )
    )
    completed = {r["trial_id"] for r in trials}
    if len(completed) != len(trials) or completed - {p[0] for p in plan}:
        raise ValueError("Duplicate or unplanned variants in checkpoint.")
    try:
        for key, instance, variant, method, seed, config in plan:
            if key in completed:
                continue
            baseline = baseline_lookup.get((instance["problem_id"], method, seed))
            if baseline is not None:
                old = baseline["algorithm_config"]
                depth_matches = (
                    old.get("p") == config.qaoa_p
                    if method == "qaoa-xy"
                    else old.get("depth") == config.binary_depth
                )
                if not depth_matches or old["alpha"] != config.alpha:
                    baseline = None
            row = measure_variant(
                key, instance, variant, method, seed, config, steps, draws, baseline
            )
            trials.append(row)
            write_json(output / "trajectories.json", trials)
            status = (
                row["error"] or f"P(target)={row['checkpoints'][-1]['p_target']:.5g}"
            )
            print(
                f"[{len(trials)}/{len(plan)}] {variant} instance={instance['instance_seed']} seed={seed} reused={row['reused_baseline']}: {status}",
                flush=True,
            )
    finally:
        export_results(output, trials, analysis, steps, draws, settings, len(plan))
    print(f"Report: {output/'report.md'}")
    return int(any(r["error"] for r in trials))


if __name__ == "__main__":
    raise SystemExit(main())
