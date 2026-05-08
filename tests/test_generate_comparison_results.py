from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_script_main():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_comparison_results.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_comparison_results", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_generate_comparison_results_classical_smoke(tmp_path):
    main = _load_script_main()
    output = tmp_path / "summary.csv"
    trials_output = tmp_path / "trials.csv"

    rc = main(
        [
            "--output",
            str(output),
            "--trials-output",
            str(trials_output),
            "--asset-counts",
            "3",
            "--methods",
            "classical",
        ]
    )

    assert rc == 0
    with output.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with trials_output.open(newline="") as f:
        trial_rows = list(csv.DictReader(f))

    assert [row["method"] for row in rows] == [
        "Classical exhaustive search",
        "Classical top-return heuristic",
        "Classical minimum-variance subset",
        "Classical equal weight",
        "Classical exact Markowitz",
    ]
    assert rows[0]["n_assets"] == "3"
    assert rows[0]["k"] == "2"
    assert rows[3]["k"] == ""
    assert len(trial_rows) == len(rows)
    assert trial_rows[0]["seed"] == "-"
