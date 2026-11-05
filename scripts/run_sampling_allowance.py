"""Derive iid sampling success curves from saved training diagnostic evidence."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from vqe_portfolio.research.infrastructure import provenance, write_csv, write_json
from vqe_portfolio.research.sampling_allowance import (
    sampling_rows,
    summarise_sampling,
    validate_draw_counts,
)
from vqe_portfolio.research.training_diagnostics import DIAGNOSTIC_METHODS

ROOT = Path(__file__).resolve().parents[1]


def plot_results(output, summaries, classical):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator

    sizes = sorted({(r["n_assets"], r["k"]) for r in classical})
    figure, axes = plt.subplots(
        1,
        len(sizes),
        figsize=(max(10, 4.2 * len(sizes)), 5.1),
        sharey=True,
        squeeze=False,
    )
    for axis, (n, k) in zip(axes[0], sizes):
        for method, color in zip(DIAGNOSTIC_METHODS, ("#2563eb", "#b45309", "#15803d")):
            rows = sorted(
                (
                    r
                    for r in summaries
                    if (r["n_assets"], r["k"], r["method"]) == (n, k, method)
                    and r["successful_pairs"]
                ),
                key=lambda r: r["draws"],
            )
            if not rows:
                continue
            for phase, style in (("initial", "--"), ("trained", "-")):
                axis.plot(
                    [r["draws"] for r in rows],
                    [100 * r[f"mean_{phase}_success_probability"] for r in rows],
                    color=color,
                    linestyle=style,
                    marker="o",
                    markersize=4,
                    label=f"{method} {phase}",
                )
        rows = sorted(
            (r for r in classical if (r["n_assets"], r["k"]) == (n, k)),
            key=lambda r: r["draws"],
        )
        axis.plot(
            [r["draws"] for r in rows],
            [100 * r["mean_success_probability"] for r in rows],
            color="#111827",
            linestyle=":",
            marker="D",
            markersize=4,
            label="uniform feasible",
        )
        counts = [r["draws"] for r in rows]
        axis.set_xscale("log", base=2)
        axis.xaxis.set_major_locator(FixedLocator(counts))
        axis.xaxis.set_major_formatter(FixedFormatter([str(m) for m in counts]))
        axis.set_title(f"{n} securities, K={k}")
        axis.set_xlabel("Independent draws")
        axis.set_ylim(-3, 103)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Chance of at least one target portfolio (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.suptitle(
        "Sampling allowance: initial circuits, trained circuits and classical sampling",
        y=0.98,
    )
    figure.text(
        0.5,
        0.90,
        "Analytical iid draws from frozen distributions; training cost excluded",
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(top=0.79, bottom=0.30, left=0.07, right=0.98, wspace=0.12)
    figure.savefig(output / "sampling_success.png", dpi=180)
    plt.close(figure)


def write_report(output, pairs, summaries, classical, source_count, draws):
    def percent(p):
        if p is None:
            return "—"
        if 0 < p < 1e-12:
            return "<1e-10%"
        return f"{100 * p:.4g}%"

    lines = [
        "# Sampling allowance diagnostic",
        "",
        f"Analysed {source_count} saved initial/trained pairs at draw allowances {', '.join(map(str, draws))}, producing {len(pairs)} paired allowance rows. No circuits were retrained and no random samples were drawn.",
        "",
        "For each saved distribution, a success is a raw draw containing a feasible binary portfolio with original-objective normalised gap <= the frozen target (0.001 in this pilot). The chance of at least one success in m independent draws with replacement is `1 - (1 - p)^m`. This is evaluated for each run before averaging over seeds within each instance and then equally over instances. Training changes are paired within each run before aggregation.",
        "",
        "Every raw quantum draw counts, including infeasible bitstrings. No feasibility conditioning, postselection or Top-K repair is applied. Uniform classical sampling draws exact-K subsets uniformly with replacement; its single-draw probability is the target subset count divided by the feasible subset count. Its baseline is evaluated once per instance and allowance, independently of quantum method/seed counts.",
        "",
        "| Securities | K | Method | Draws | Initial success | Trained success | Change (percentage points) | Uniform feasible (matched coverage) | Valid pairs / all |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summaries:
        delta = r["mean_training_change"]
        change = "—" if delta is None else f"{100 * delta:+.4g}"
        lines.append(
            f"| {r['n_assets']} | {r['k']} | {r['method']} | {r['draws']} | {percent(r['mean_initial_success_probability'])} | {percent(r['mean_trained_success_probability'])} | {change} | {percent(r['mean_uniform_feasible_success_probability'])} | {r['successful_pairs']} / {r['pair_count']} |"
        )
    lines += [
        "",
        "## Uniform feasible baseline across all source instances",
        "",
        "| Securities | K | Draws | Instances | Success probability |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in classical:
        lines.append(
            f"| {r['n_assets']} | {r['k']} | {r['draws']} | {r['instance_count']} | {percent(r['mean_success_probability'])} |"
        )
    lines += [
        "",
        "Probabilities are analytical expectations, not observed hit fractions or new stochastic trials. In particular, a mean single-draw probability is never substituted into the nonlinear success formula. Near-100% entries may round to 100%; full precision is retained in CSVs. Values below 1e-10% are displayed as an upper bound. Markers in the plot show the evaluated allowances; connecting lines guide the eye.",
        "",
        "Complete source checkpoints are required. Failed source pairs remain as blank probabilities and error counts; probability means condition on valid source pairs, with quality_instance_count and successful_pairs reporting coverage. The matched classical column uses the same successful-pair coverage, while baseline_summary.csv uses all source instances. With failures, model and all-instance baseline curves can have different coverage and must be interpreted conditionally.",
        "",
        "The formula assumes a fixed distribution and independent draws with replacement. It does not model device noise, drift or adaptive sampling. Draw counts do not equalise runtime or cost: training, circuit preparation, measurements and classical candidate processing are excluded. This is a sampling-reliability comparison, not a quantum-advantage or time-to-solution result. The pilot contains only three independent instances per size and three initialisations per method; no confidence intervals or scaling claims are made.",
        "",
        "manifest.json hashes both source evidence files and records the analysis settings, software and source versions. pairs.csv preserves per-run allowance probabilities and paired changes. baseline_trials.csv contains one classical row per instance/allowance. summary.csv and baseline_summary.csv contain descriptive means.",
        "",
        "![Success probability by sampling allowance](sampling_success.png)",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-dir", type=Path, default=ROOT / "results/training_diagnostic"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/sampling_allowance"
    )
    parser.add_argument(
        "--draws", default="16,64,512", help="Comma-separated positive draw counts."
    )
    args = parser.parse_args(argv)
    draws = validate_draw_counts(tuple(int(m) for m in args.draws.split(",")))
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is nonempty; use a new directory.")
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
    records = json.loads(inputs["pairs.json"])
    pairs, baselines = sampling_rows(source, records, draws)
    summaries, classical = summarise_sampling(pairs, baselines)
    environment = provenance(ROOT, Path(__file__), extra_packages=("matplotlib",))
    manifest = {
        "schema_version": 1,
        "study": "iid_sampling_allowance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "draw_counts": list(draws),
        "target_gap": source["config"]["target_gap"],
        "source_pair_count": len(records),
        "source_instance_count": len(source["instances"]),
        "source_inputs_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in inputs.items()
        },
        "environment": environment,
        "training_performed": False,
        "sampling_performed": False,
        "assumption": "independent fixed-distribution draws with replacement; every raw quantum draw counts",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    for filename, rows in (
        ("pairs.csv", pairs),
        ("baseline_trials.csv", baselines),
        ("summary.csv", summaries),
        ("baseline_summary.csv", classical),
    ):
        write_csv(output / filename, rows)
    plot_results(output, summaries, classical)
    write_report(output, pairs, summaries, classical, len(records), draws)
    print(
        f"Analysed {len(records)} pairs at {len(draws)} allowances; source errors: {sum(bool(r['error']) for r in records)}"
    )
    print(f"Report: {output / 'report.md'}")
    return int(any(r["error"] for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
