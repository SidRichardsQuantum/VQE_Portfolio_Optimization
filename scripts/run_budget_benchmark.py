"""Reproduce fixed-deadline simulator benchmarks, retaining every failed run."""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqe_portfolio.execution import BUDGET_METHODS  # noqa: E402
from vqe_portfolio.problem import PortfolioProblem  # noqa: E402
from vqe_portfolio.research.budget import (  # noqa: E402
    BUDGET_VARIANTS,
    evaluate_budget_trial,
    paired_budget_trials,
    summarise_budget_trials,
)
from vqe_portfolio.research.reliability import variant_config  # noqa: E402
from vqe_portfolio.research.infrastructure import (  # noqa: E402
    provenance,
    write_csv,
    write_json,
)
from vqe_portfolio.research.validation import (  # noqa: E402
    ValidationConfig,
    prepare_instances,
)
from vqe_portfolio.types import (  # noqa: E402
    BinaryVQEConfig,
    FractionalVQEConfig,
    QAOAConfig,
)


def timing_report(trials):
    from collections import defaultdict
    from statistics import mean

    groups = defaultdict(list)
    for row in trials:
        groups[(row["variant"], row["budget_seconds"])].append(row["timings"])
    lines = [
        "",
        "## Timing diagnostics",
        "",
        "Worker phase durations include the active phase up to termination. Startup includes spawn and imports. Missing startup completion is censored at the deadline. Delivery and parent validation overlap worker time and are not additive phases. See timings.csv for per-trial measurements and interrupted phases.",
        "",
        "All durations below are seconds. Delivery is the mean over delivered incumbents, averaged over runs with delivery; a dash means none arrived. Parent validation and cleanup means include all runs.",
        "",
        "| Variant | Budget | Startup | Setup | Optimization | Candidates | Delivery | Validation | Cleanup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (variant, seconds), rows in groups.items():
        values = [
            mean(r[f"{p}_seconds"] for r in rows)
            for p in (
                "startup",
                "setup",
                "optimization",
                "candidate_processing",
            )
        ]
        deliveries = [
            r["mean_delivery_seconds"]
            for r in rows
            if r["mean_delivery_seconds"] is not None
        ]
        delivery = f"{mean(deliveries):.4f}" if deliveries else "—"
        lines.append(
            f"| {variant} | {seconds:g} | "
            + " | ".join(f"{v:.4f}" for v in values)
            + f" | {delivery} | {mean(r['parent_evaluation_seconds'] for r in rows):.4f} | {mean(r['cleanup_seconds'] for r in rows):.4f} |"
        )
    lines += [
        "",
        "## Fresh-instance and sampling controls",
        "",
        "Excluded development manifests and their hashes are frozen in manifest.json; overlap is rejected before solving. Named presets freeze the v0.3.1 reliability settings. Trained/untrained Dicke-ring variants share production parameter initialization for each seed. The trained variant publishes candidates during training, then spends the remaining allowance repeatedly sampling its final parameters. The untrained variant repeatedly executes the initial circuit with zero Adam updates and actual finite-shot batches. Uniform sampling and both circuit sampling loops may stop once every feasible exact-K portfolio has actually been sampled, using only n and K, never a reference target. Otherwise sampling uses the full allowance. Finite training caps and early completion are explicit algorithm settings; equal allowances do not imply equal time spent.",
        "",
        "pairs.csv compares the trained Dicke-ring variant with each binary comparator on identical instance/seed/budget keys. Positive normalized improvement favors training. Quality differences require both feasible returns; failure and target-hit flags remain explicit. These are descriptive pairs, without confidence intervals or a formal held-out tuning study.",
    ]
    return lines


def paired_report(trials):
    from collections import defaultdict
    from statistics import mean

    groups = defaultdict(list)
    for row in paired_budget_trials(trials):
        if row["budget_seconds"] == 10:
            groups[(row["n_assets"], row["comparator"])].append(row)
    if not groups:
        return []
    lines = [
        "",
        "## Paired comparisons at the primary 10-second allowance",
        "",
        "The trained variant is Dicke-ring XY QAOA. Positive mean normalized improvement favors training. Gap means use only pairs with two feasible returns; target-hit rates include every pair. Each row reports observed outcomes, without uncertainty estimates.",
        "",
        "| Assets | Comparator | Pairs | Both feasible | Trained target hit | Comparator target hit | Mean improvement |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (n, comparator), rows in groups.items():
        gaps = [r["normalised_improvement"] for r in rows if r["both_feasible"]]
        gap = f"{mean(gaps):.6g}" if gaps else "—"
        lines.append(
            f"| {n} | {comparator} | {len(rows)} | {len(gaps)} | {mean(r['trained_target_hit'] for r in rows):.1%} | {mean(r['comparator_target_hit'] for r in rows):.1%} | {gap} |"
        )
    return lines


def solver_config(method, instance, seed, args):
    if method in BUDGET_VARIANTS:
        problem = PortfolioProblem.from_dict(instance["snapshot"])
        preset = {
            "binary-improved": "binary-restarts",
            "xy-dicke-ring-trained": "xy-dicke-ring",
            "xy-dicke-ring-untrained": "xy-dicke-ring",
        }.get(method, method)
        cfg = variant_config(preset, problem, seed, args.steps, args.shots, 3)
        return replace(
            cfg,
            log_every=args.checkpoint_every,
            **({"steps": 0} if method.endswith("-untrained") else {}),
        )
    common = dict(
        lam=instance["snapshot"]["lam"],
        seed=seed,
        steps=args.steps,
        log_every=args.checkpoint_every,
        stepsize=args.stepsize,
    )
    if method == "fractional-vqe":
        return FractionalVQEConfig(**common)
    if method not in {
        "binary-vqe",
        "qaoa-x",
        "qaoa-xy",
        "qaoa-xy-untrained",
        "qaoa-xy-trained",
    }:
        return None
    common.update(
        k=instance["snapshot"]["k"], alpha=args.alpha, shots_sample=args.shots
    )
    if method == "binary-vqe":
        return BinaryVQEConfig(
            **common,
            restarts=args.restarts,
            normalize_energy=args.normalize_energy,
            keep_best=args.keep_best,
            stepsize_decay=args.stepsize_decay,
        )
    xy = method.startswith("qaoa-xy")
    if method == "qaoa-xy-untrained":
        common["steps"] = 0
    return QAOAConfig(
        **common,
        p=args.qaoa_p,
        mixer="xy" if xy else "x",
        xy_initial_state=args.xy_initial_state if xy else "basis",
        xy_topology=args.xy_topology if xy else "ring",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="JSON argument defaults; CLI overrides them"
    )
    parser.add_argument("--asset-counts", default="4,6,8")
    parser.add_argument("--instance-seeds", default="101,202,303")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default=",".join(BUDGET_METHODS))
    parser.add_argument(
        "--variants",
        default="",
        help="Named presets and/or method names, comma-separated",
    )
    parser.add_argument(
        "--exclude-manifests",
        default="",
        help="Prior study manifests whose problem IDs must not recur",
    )
    parser.add_argument("--budgets", default="1,10,60")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--stepsize", type=float, default=0.3)
    parser.add_argument("--stepsize-decay", type=float, default=0.0)
    parser.add_argument("--alpha", default="5.0")
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--normalize-energy", action="store_true")
    parser.add_argument("--keep-best", action="store_true")
    parser.add_argument("--qaoa-p", type=int, default=1)
    parser.add_argument(
        "--xy-initial-state", choices=("basis", "dicke"), default="basis"
    )
    parser.add_argument("--xy-topology", choices=("ring", "complete"), default="ring")
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--lam", type=float, default=4.0)
    parser.add_argument("--target-gap", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/budget_benchmark"
    )
    parser.add_argument("--resume", action="store_true")
    preliminary, _ = parser.parse_known_args(argv)
    config_document = None
    if preliminary.config:
        config_document = json.loads(preliminary.config.read_text())
        allowed = {a.dest for a in parser._actions} - {
            "help",
            "config",
            "resume",
            "output_dir",
        }
        if set(config_document) - allowed:
            parser.error("Unknown or non-study setting in configuration")
        parser.set_defaults(**config_document)
    args = parser.parse_args(argv)
    args.alpha = "auto" if args.alpha == "auto" else float(args.alpha)

    def ints(s):
        return tuple(int(x) for x in s.split(","))

    variants = tuple((args.variants or args.methods).split(","))
    if (
        not variants
        or len(set(variants)) != len(variants)
        or set(variants) - (BUDGET_METHODS.keys() | BUDGET_VARIANTS.keys())
    ):
        parser.error("Use distinct known methods or variants")
    dispatch = {v: BUDGET_VARIANTS.get(v, v) for v in variants}
    if "binary-improved" in variants and (args.steps < 3 or args.steps % 3):
        parser.error("binary-improved needs total steps to be a positive multiple of 3")
    methods = tuple(dict.fromkeys(dispatch.values()))
    budgets = tuple(float(x) for x in args.budgets.split(","))
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or any(not math.isfinite(x) or x <= 0 for x in budgets)
    ):
        parser.error("Use distinct finite positive budgets")
    if (
        args.checkpoint_every < 1
        or args.restarts < 1
        or args.qaoa_p < 1
        or not math.isfinite(args.stepsize)
        or args.stepsize <= 0
        or not math.isfinite(args.stepsize_decay)
        or args.stepsize_decay < 0
    ):
        parser.error("Invalid solver settings")
    config = ValidationConfig(
        asset_counts=ints(args.asset_counts),
        instance_seeds=ints(args.instance_seeds),
        solver_seeds=ints(args.seeds),
        methods=tuple(
            dict.fromkeys("qaoa-xy" if m.startswith("qaoa-xy-") else m for m in methods)
        ),
        steps=args.steps,
        shots=args.shots,
        lam=args.lam,
        target_gap=args.target_gap,
    )
    settings = {
        k: v
        for k, v in vars(args).items()
        if k not in {"output_dir", "resume", "config"}
    }
    settings["config_document"] = config_document
    excluded = []
    excluded_ids = set()
    for name in filter(None, args.exclude_manifests.split(",")):
        path = ROOT / name
        raw = path.read_bytes()
        prior = json.loads(raw)
        ids = {i["problem_id"] for i in prior["instances"]}
        excluded_ids.update(ids)
        excluded.append(
            {
                "path": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "problem_ids": sorted(ids),
            }
        )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    environment = provenance(ROOT, Path(__file__))
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["settings"] != settings or manifest["environment"] != environment:
            raise ValueError("Resume refused: settings, code or environment changed")
        trials = json.loads((output / "trials.json").read_text())
        if manifest["excluded_studies"] != excluded:
            raise ValueError("Resume refused: excluded studies changed")
    else:
        if any(output.iterdir()):
            raise ValueError("Use an empty output directory")
        manifest = {
            "schema_version": 2,
            "study": "fixed_deadline_simulator_pilot",
            "settings": settings,
            "environment": environment,
            "instances": prepare_instances(config),
            "excluded_studies": excluded,
            "timing": "Spawn startup, setup, encoding, training, sampling, repair, evaluation and delivery included; cleanup overrun reported separately.",
        }
        if excluded_ids & {i["problem_id"] for i in manifest["instances"]}:
            raise ValueError(
                "Fresh-instance study overlaps excluded development problems"
            )
        write_json(output / "manifest.json", manifest)
        trials = []
        write_json(output / "trials.json", trials)
    plan = []
    for instance in manifest["instances"]:
        for variant, method in dispatch.items():
            if BUDGET_METHODS[method] == instance["snapshot"]["objective_family"]:
                for seed in config.solver_seeds:  # repeats deterministic timings too
                    for seconds in budgets:
                        key = f"{instance['problem_id']}:{variant}:{seed}:{seconds:g}"
                        plan.append((key, instance, variant, method, seed, seconds))
    completed = {r["trial_id"] for r in trials}
    if len(completed) != len(trials) or completed - {p[0] for p in plan}:
        raise ValueError("Duplicate or unplanned trial in checkpoint")
    try:
        for key, instance, variant, method, seed, seconds in plan:
            if key in completed:
                continue
            cfg = solver_config(variant, instance, seed, args)
            row = evaluate_budget_trial(
                instance, method, seed, seconds, cfg, args.target_gap, variant=variant
            )
            trials.append(row)
            write_json(output / "trials.json", trials)
            print(
                f"[{len(trials)}/{len(plan)}] n={row['n_assets']} {variant} {seconds:g}s: {row['status']}, target={row['target_hit']}",
                flush=True,
            )
    finally:
        summary, reliability = summarise_budget_trials(trials)
        write_csv(
            output / "trials.csv",
            trials,
            json_fields=("candidate", "incumbents", "solver_config", "timings"),
        )
        write_csv(output / "summary.csv", summary)
        write_csv(output / "reliability.csv", reliability)
        write_csv(output / "pairs.csv", paired_budget_trials(trials))
        write_csv(
            output / "timings.csv",
            [
                {
                    "trial_id": r["trial_id"],
                    "problem_id": r["problem_id"],
                    "n_assets": r["n_assets"],
                    "variant": r["variant"],
                    "seed": r["seed"],
                    "budget_seconds": r["budget_seconds"],
                    "status": r["status"],
                    **r["timings"],
                }
                for r in trials
            ],
        )
        lines = [
            "# Fixed-deadline simulator pilot",
            "",
            f"Completed {len(trials)} of {len(plan)} trials. Every timeout and error remains in the denominators.",
            "",
            "Spawn startup and all method-specific work consume the budget. Only portfolios received before the deadline are scored. Cleanup overruns are retained in the trial records. No shared warm worker or amortization is assumed.",
            "",
            "| Assets | Variant | Seconds | Feasible | Target hit | Errors |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for r in summary:
            lines.append(
                f"| {r['n_assets']} | {r['variant']} | {r['budget_seconds']:g} | {r['feasible_rate']:.1%} | {r['target_hit_rate']:.1%} | {r['error_count']} |"
            )
        lines += paired_report(trials)
        lines += timing_report(trials)
        lines += [
            "",
            "Quality means condition on feasible returns, with coverage in summary.csv. reliability.csv reports per-instance empirical T90 using all runs, including censored failures; a blank means unreached within the budget. Reference computation is excluded. Binary and fractional objectives remain separate. These are descriptive simulator results, without confidence intervals, held-out tuning, a scaling claim, or hardware advantage evidence.",
            "",
        ]
        (output / "report.md").write_text("\n".join(lines))
    return int(any(r["error"] or r["reference_contradiction"] for r in trials))


if __name__ == "__main__":
    raise SystemExit(main())
