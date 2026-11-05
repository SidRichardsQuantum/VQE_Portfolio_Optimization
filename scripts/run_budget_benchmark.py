"""Reproduce fixed-deadline simulator benchmarks, retaining every failed run."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vqe_portfolio.execution import BUDGET_METHODS  # noqa: E402
from vqe_portfolio.research.budget import (  # noqa: E402
    evaluate_budget_trial,
    summarise_budget_trials,
)
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


def solver_config(method, instance, seed, args):
    common = dict(
        lam=instance["snapshot"]["lam"],
        seed=seed,
        steps=args.steps,
        log_every=args.checkpoint_every,
        stepsize=args.stepsize,
    )
    if method == "fractional-vqe":
        return FractionalVQEConfig(**common)
    if method not in {"binary-vqe", "qaoa-x", "qaoa-xy"}:
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
    xy = method == "qaoa-xy"
    return QAOAConfig(
        **common,
        p=args.qaoa_p,
        mixer="xy" if xy else "x",
        xy_initial_state=args.xy_initial_state if xy else "basis",
        xy_topology=args.xy_topology if xy else "ring",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-counts", default="4,6,8")
    parser.add_argument("--instance-seeds", default="101,202,303")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default=",".join(BUDGET_METHODS))
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
    args = parser.parse_args(argv)
    args.alpha = "auto" if args.alpha == "auto" else float(args.alpha)

    def ints(s):
        return tuple(int(x) for x in s.split(","))

    methods = tuple(args.methods.split(","))
    if set(methods) - BUDGET_METHODS.keys():
        parser.error("Unknown method")
    budgets = tuple(float(x) for x in args.budgets.split(","))
    import math

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
        or args.stepsize <= 0
        or args.stepsize_decay < 0
    ):
        parser.error("Invalid solver settings")
    config = ValidationConfig(
        asset_counts=ints(args.asset_counts),
        instance_seeds=ints(args.instance_seeds),
        solver_seeds=ints(args.seeds),
        methods=methods,
        steps=args.steps,
        shots=args.shots,
        lam=args.lam,
        target_gap=args.target_gap,
    )
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
        manifest = {
            "schema_version": 1,
            "study": "fixed_deadline_simulator_pilot",
            "settings": settings,
            "environment": environment,
            "instances": prepare_instances(config),
            "timing": "Spawn startup, setup, encoding, training, sampling, repair, evaluation and delivery included; cleanup overrun reported separately.",
        }
        write_json(output / "manifest.json", manifest)
        trials = []
        write_json(output / "trials.json", trials)
    plan = []
    for instance in manifest["instances"]:
        for method in methods:
            if BUDGET_METHODS[method] == instance["snapshot"]["objective_family"]:
                for seed in config.solver_seeds:  # repeats deterministic timings too
                    for seconds in budgets:
                        key = f"{instance['problem_id']}:{method}:{seed}:{seconds:g}"
                        plan.append((key, instance, method, seed, seconds))
    completed = {r["trial_id"] for r in trials}
    if len(completed) != len(trials) or completed - {p[0] for p in plan}:
        raise ValueError("Duplicate or unplanned trial in checkpoint")
    try:
        for key, instance, method, seed, seconds in plan:
            if key in completed:
                continue
            cfg = solver_config(method, instance, seed, args)
            row = evaluate_budget_trial(
                instance, method, seed, seconds, cfg, args.target_gap
            )
            trials.append(row)
            write_json(output / "trials.json", trials)
            print(
                f"[{len(trials)}/{len(plan)}] {method} {seconds:g}s: {row['status']}, target={row['target_hit']}",
                flush=True,
            )
    finally:
        summary, reliability = summarise_budget_trials(trials)
        write_csv(
            output / "trials.csv",
            trials,
            json_fields=("candidate", "incumbents", "solver_config"),
        )
        write_csv(output / "summary.csv", summary)
        write_csv(output / "reliability.csv", reliability)
        lines = [
            "# Fixed-deadline simulator pilot",
            "",
            f"Completed {len(trials)} of {len(plan)} trials. Every timeout and error remains in the denominators.",
            "",
            "Spawn startup and all method-specific work consume the budget. Only portfolios received before the deadline are scored. Cleanup overruns are retained in the trial records. No shared warm worker or amortization is assumed.",
            "",
            "| Assets | Method | Seconds | Feasible | Target hit | Errors |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for r in summary:
            lines.append(
                f"| {r['n_assets']} | {r['method']} | {r['budget_seconds']:g} | {r['feasible_rate']:.1%} | {r['target_hit_rate']:.1%} | {r['error_count']} |"
            )
        lines += [
            "",
            "Quality means condition on feasible returns, with coverage in summary.csv. reliability.csv reports per-instance empirical T90 using all runs, including censored failures; a blank means unreached within the budget. Reference computation is excluded. Binary and fractional objectives remain separate. These are descriptive simulator results, without confidence intervals, held-out tuning, a scaling claim, or hardware advantage evidence.",
            "",
        ]
        (output / "report.md").write_text("\n".join(lines))
    return int(any(r["error"] or r["reference_contradiction"] for r in trials))


if __name__ == "__main__":
    raise SystemExit(main())
