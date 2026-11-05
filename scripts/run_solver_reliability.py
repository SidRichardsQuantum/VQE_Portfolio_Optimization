"""Compare solver reliability controls on frozen development instances."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqe_portfolio.research.infrastructure import (  # noqa: E402
    provenance,
    write_csv,
    write_json,
)
from vqe_portfolio.research.reliability import VARIANTS, measure_variant  # noqa: E402
from vqe_portfolio.research.validation import (  # noqa: E402
    ValidationConfig,
    prepare_instances,
)


def export(output, trials, planned):
    groups = defaultdict(lambda: defaultdict(list))
    for row in trials:
        groups[(row["n_assets"], row["variant"])][row["problem_id"]].append(row)
    summary = []
    for (n, variant), instances in groups.items():
        all_rows = [r for rows in instances.values() for r in rows]
        row = dict(
            n_assets=n,
            variant=variant,
            instance_count=len(instances),
            trial_count=len(all_rows),
            error_count=sum(bool(r["error"]) for r in all_rows),
        )
        for metric in (
            "p_target",
            "p_feasible",
            "success_16",
            "success_64",
            "success_512",
            "uniform_success_64",
        ):
            values = [
                mean(r[metric] for r in rows if not r["error"])
                for rows in instances.values()
                if any(not r["error"] for r in rows)
            ]
            row[metric] = mean(values) if values else None
        row["quality_instance_count"] = sum(
            any(not r["error"] for r in rows) for rows in instances.values()
        )
        summary.append(row)
    write_csv(
        output / "trials.csv", trials, json_fields=("config", "params", "probabilities")
    )
    write_csv(output / "summary.csv", summary)
    lines = [
        "# Solver reliability development study",
        "",
        f"Completed {len(trials)} of {planned} planned trials.",
        "",
        "Binary variants compare the legacy penalty, a reference-free sufficient auto penalty, normalization, and a combined restart/best-iterate/learning-rate-decay strategy. Each variant receives the same total number of Adam updates; restarts divide those updates equally. They do not have equal wall-clock or circuit-evaluation budgets. XY variants compare basis and uniform exact-K (Dicke) initial states with ring and complete ordered exchange networks at depth 1.",
        "",
        "All settings and problem references are frozen in manifest.json. Optimizers never receive reference scores or target sets. Exact noiseless target probabilities are evaluated after solving; they are distinct from the sampled candidate hit rate. Success at m draws is computed per run before averaging seeds within instances and giving instances equal weight.",
        "",
        "| Assets | Variant | Target per draw | Feasible | Success: 64 draws | Uniform exact-K: 64 draws | Errors |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt(value):
        return "—" if value is None else f"{value:.3%}"

    for r in summary:
        lines.append(
            f"| {r['n_assets']} | {r['variant']} | {fmt(r['p_target'])} | {fmt(r['p_feasible'])} | {fmt(r['success_64'])} | {fmt(r['uniform_success_64'])} | {r['error_count']} |"
        )
    lines += [
        "",
        "Errors remain in trial counts; probability means condition on successful runs and report instance coverage. The restart variant changes several controls together, so its outcome does not identify the contribution of any one control. StatePrep uses an explicit 2**n amplitude vector for the Dicke state; this is a small-instance simulator implementation, without a hardware preparation-cost claim. Complete networks use O(n**2) exchange edges. These are development instances with no held-out tuning, confidence intervals, scaling law, or quantum-advantage claim. Historical evidence is unchanged.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-counts", default="4,6,8")
    parser.add_argument("--instance-seeds", default="101,202,303")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/solver_reliability"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    def ints(text):
        return tuple(int(x) for x in text.split(","))

    config = ValidationConfig(
        asset_counts=ints(args.asset_counts),
        instance_seeds=ints(args.instance_seeds),
        solver_seeds=ints(args.seeds),
        methods=("binary-vqe",),
        steps=args.steps,
        shots=args.shots,
    )
    if args.restarts < 1 or args.steps < args.restarts or args.steps % args.restarts:
        parser.error("steps must be a positive multiple of restarts")
    settings = {
        k: v for k, v in vars(args).items() if k not in {"output_dir", "resume"}
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    environment = provenance(ROOT, Path(__file__))
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["settings"] != settings or manifest["environment"] != environment:
            raise ValueError("Resume refused: settings, code or environment changed")
        trials = json.loads((output / "trials.json").read_text())
    else:
        if any(output.iterdir()):
            raise ValueError("Use an empty output directory")
        manifest = dict(
            schema_version=1,
            study="solver_reliability_development",
            settings=settings,
            environment=environment,
            instances=prepare_instances(config),
            variants=list(VARIANTS),
        )
        write_json(output / "manifest.json", manifest)
        trials = []
        write_json(output / "trials.json", trials)
    plan = [
        (f"{i['problem_id']}:{v}:{seed}", i, v, seed)
        for i in manifest["instances"]
        for v in VARIANTS
        for seed in config.solver_seeds
    ]
    completed = {r["trial_id"] for r in trials}
    if len(completed) != len(trials) or completed - {p[0] for p in plan}:
        raise ValueError("Duplicate or unplanned trial in checkpoint")
    try:
        for key, instance, variant, seed in plan:
            if key in completed:
                continue
            row = measure_variant(
                instance, variant, seed, args.steps, args.shots, args.restarts
            )
            trials.append(row)
            write_json(output / "trials.json", trials)
            print(
                f"[{len(trials)}/{len(plan)}] n={row['n_assets']} {variant}: {row.get('p_target', row['error'])}",
                flush=True,
            )
    finally:
        export(output, trials, len(plan))
    return int(any(r["error"] for r in trials))


if __name__ == "__main__":
    raise SystemExit(main())
