from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_script_main():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_ansatz_comparison.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_ansatz_comparison", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_generate_ansatz_comparison_smoke(tmp_path):
    main = _load_script_main()
    output = tmp_path / "ansatz.csv"

    rc = main(
        [
            "--output",
            str(output),
            "--asset-count",
            "3",
            "--seeds",
            "0",
            "--depth",
            "1",
            "--steps",
            "1",
            "--shots",
            "32",
            "--skip-plots",
        ]
    )

    assert rc == 0
    with output.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert [row["ansatz"] for row in rows] == [
        "-",
        "ry_cz",
        "ry_rz_cz",
        "strongly_entangling",
        "-",
        "ry",
        "ry_cz",
        "ry_rz_cz",
    ]
    assert rows[1]["method"] == "Binary VQE"
    assert rows[5]["method"] == "Fractional VQE"
