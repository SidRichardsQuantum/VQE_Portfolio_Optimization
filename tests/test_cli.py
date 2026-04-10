from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _run(args):
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = str(repo_root / "src")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{src_dir}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_dir
    )
    return subprocess.run(
        [sys.executable, "-m", "vqe_portfolio", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_help():
    p = _run(["--help"])
    assert p.returncode == 0
    assert "usage" in p.stdout.lower()


def test_cli_binary_smoke():
    p = _run(
        [
            "binary",
            "--mu",
            "0.1,0.2",
            "--sigma",
            "0.1,0.0;0.0,0.2",
            "--k",
            "1",
            "--lam",
            "0.5",
            "--steps",
            "2",
            "--depth",
            "1",
            "--shots-sample",
            "50",
        ]
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(p.stdout)
    assert payload["method"] == "binary"


def test_cli_fractional_smoke():
    p = _run(
        [
            "fractional",
            "--mu",
            "0.1,0.2",
            "--sigma",
            "0.1,0.0;0.0,0.2",
            "--lam",
            "0.5",
            "--steps",
            "2",
            "--shots",
            "50",
        ]
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(p.stdout)
    assert payload["method"] == "fractional"


def test_cli_binary_data_help():
    p = _run(["binary-data", "--help"])
    assert p.returncode == 0


def test_cli_fractional_data_help():
    p = _run(["fractional-data", "--help"])
    assert p.returncode == 0


def test_cli_binary_requires_input_or_mu_sigma():
    p = _run(["binary", "--k", "1", "--lam", "0.5"])

    assert p.returncode == 2
    assert "Provide either --input JSON or both --mu and --sigma." in p.stderr


def test_cli_binary_rejects_dimension_mismatch():
    p = _run(
        [
            "binary",
            "--mu",
            "0.1,0.2",
            "--sigma",
            "0.1,0.0,0.0;0.0,0.2,0.0;0.0,0.0,0.3",
            "--k",
            "1",
            "--lam",
            "0.5",
        ]
    )

    assert p.returncode == 2
    assert "Dimension mismatch" in p.stderr


def test_cli_binary_rejects_k_out_of_range():
    p = _run(
        [
            "binary",
            "--mu",
            "0.1,0.2",
            "--sigma",
            "0.1,0.0;0.0,0.2",
            "--k",
            "3",
            "--lam",
            "0.5",
            "--steps",
            "1",
            "--depth",
            "1",
            "--shots-sample",
            "10",
        ]
    )

    assert p.returncode == 2
    assert "cfg.k must be in [1, 2] but got 3" in p.stderr


def test_cli_qaoa_invalid_mixer_rejected_by_parser():
    p = _run(
        [
            "qaoa",
            "--mu",
            "0.1,0.2",
            "--sigma",
            "0.1,0.0;0.0,0.2",
            "--k",
            "1",
            "--lam",
            "0.5",
            "--mixer",
            "bad",
        ]
    )

    assert p.returncode != 0
    assert "invalid choice" in p.stderr.lower()


def test_cli_binary_rejects_invalid_json_input(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"mu": [0.1, 0.2], "sigma": [}')

    p = _run(["binary", "--input", str(bad_json), "--k", "1", "--lam", "0.5"])

    assert p.returncode == 2
    assert "error:" in p.stderr


def test_cli_fractional_writes_output_file(tmp_path):
    out_path = tmp_path / "fractional.json"

    p = _run(
        [
            "fractional",
            "--mu",
            "0.1,0.2",
            "--sigma",
            "0.1,0.0;0.0,0.2",
            "--lam",
            "0.5",
            "--steps",
            "2",
            "--shots",
            "50",
            "--out",
            str(out_path),
        ]
    )

    assert p.returncode == 0, p.stderr
    assert p.stdout == ""

    payload = json.loads(out_path.read_text())
    assert payload["method"] == "fractional"
