from __future__ import annotations

import json

from vqe_portfolio.cli import main


def test_cli_qaoa_stdout(capsys):
    argv = [
        "qaoa",
        "--mu",
        "0.10,0.20,0.15",
        "--sigma",
        "0.05,0.01,0.00;0.01,0.06,0.01;0.00,0.01,0.04",
        "--k",
        "2",
        "--lam",
        "1.0",
        "--alpha",
        "5.0",
        "--p",
        "1",
        "--steps",
        "2",
        "--stepsize",
        "0.1",
        "--shots-sample",
        "64",
        "--seed",
        "0",
        "--mixer",
        "x",
    ]

    rc = main(argv)
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)

    assert payload["method"] == "qaoa"

    cfg = payload["config"]
    assert cfg["p"] == 1
    assert cfg["k"] == 2
    assert cfg["lam"] == 1.0
    assert cfg["alpha"] == 5.0
    assert cfg["mixer"] == "x"

    res = payload["result"]
    assert "params" in res
    assert "gammas" in res
    assert "betas" in res
    assert "cost_trace" in res
    assert "state_probs" in res
    assert "x_prob" in res
    assert "x_round" in res
    assert "x_topk" in res
    assert "sample_counts" in res
    assert "x_mode" in res
    assert "x_best_feasible" in res


def test_cli_qaoa_xy_stdout(capsys):
    argv = [
        "qaoa",
        "--mu",
        "0.10,0.20,0.15",
        "--sigma",
        "0.05,0.01,0.00;0.01,0.06,0.01;0.00,0.01,0.04",
        "--k",
        "2",
        "--lam",
        "1.0",
        "--alpha",
        "5.0",
        "--p",
        "1",
        "--steps",
        "2",
        "--stepsize",
        "0.1",
        "--shots-sample",
        "64",
        "--seed",
        "0",
        "--mixer",
        "xy",
    ]

    rc = main(argv)
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)

    assert payload["method"] == "qaoa"
    assert payload["config"]["mixer"] == "xy"


def test_cli_qaoa_outfile(tmp_path):
    out_path = tmp_path / "qaoa_result.json"

    argv = [
        "qaoa",
        "--mu",
        "0.10,0.20,0.15",
        "--sigma",
        "0.05,0.01,0.00;0.01,0.06,0.01;0.00,0.01,0.04",
        "--k",
        "2",
        "--lam",
        "1.0",
        "--alpha",
        "5.0",
        "--p",
        "1",
        "--steps",
        "2",
        "--stepsize",
        "0.1",
        "--shots-sample",
        "64",
        "--seed",
        "0",
        "--out",
        str(out_path),
    ]

    rc = main(argv)
    assert rc == 0
    assert out_path.exists()

    payload = json.loads(out_path.read_text())
    assert payload["method"] == "qaoa"
    assert payload["config"]["p"] == 1


def test_cli_qaoa_requires_input(capsys):
    argv = [
        "qaoa",
        "--k",
        "2",
        "--lam",
        "1.0",
        "--alpha",
        "5.0",
        "--p",
        "1",
    ]

    rc = main(argv)
    assert rc == 2
    assert (
        "Provide either --input JSON or both --mu and --sigma."
        in capsys.readouterr().err
    )


def test_cli_qaoa_accepts_input_json(tmp_path, capsys):
    in_path = tmp_path / "inputs.json"
    in_path.write_text(
        json.dumps(
            {
                "mu": [0.10, 0.20, 0.15],
                "sigma": [
                    [0.05, 0.01, 0.00],
                    [0.01, 0.06, 0.01],
                    [0.00, 0.01, 0.04],
                ],
            }
        )
    )

    argv = [
        "qaoa",
        "--input",
        str(in_path),
        "--k",
        "2",
        "--lam",
        "1.0",
        "--alpha",
        "5.0",
        "--p",
        "1",
        "--steps",
        "2",
        "--stepsize",
        "0.1",
        "--shots-sample",
        "64",
        "--seed",
        "0",
    ]

    rc = main(argv)
    assert rc == 0

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["method"] == "qaoa"
    assert payload["config"]["k"] == 2
