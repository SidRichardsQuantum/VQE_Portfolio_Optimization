from __future__ import annotations

import json
import subprocess
import sys


def _run(args):
    return subprocess.run([sys.executable, "-m", "vqe_portfolio", *args], capture_output=True, text=True)


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
