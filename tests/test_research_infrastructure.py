import csv
import hashlib
import json

import pytest

from vqe_portfolio.research.infrastructure import provenance, write_csv, write_json


def test_provenance_tracks_nested_sources_and_explicit_runner(tmp_path):
    package = tmp_path / "src/vqe_portfolio"
    nested = package / "research"
    nested.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    source = nested / "study.py"
    source.write_text("STEPS = 25\n")
    runner = tmp_path / "runner.py"
    runner.write_text("# study runner\n")
    before = provenance(tmp_path, runner, extra_packages=("matplotlib",))
    assert before == provenance(tmp_path, runner, extra_packages=("matplotlib",))
    hashes = before["source_sha256"]
    assert set(hashes) == {
        "src/vqe_portfolio/__init__.py",
        "src/vqe_portfolio/research/study.py",
        "runner.py",
    }
    assert hashes["runner.py"] == hashlib.sha256(runner.read_bytes()).hexdigest()
    assert "matplotlib" in before["packages"]
    source.write_text("STEPS = 100\n")
    after = provenance(tmp_path, runner, extra_packages=("matplotlib",))
    assert after != before  # A nested source change invalidates resume identity.
    assert after["source_sha256"]["runner.py"] == hashes["runner.py"]


def test_checkpoint_replacement_rejects_nonfinite_data(tmp_path):
    path = tmp_path / "trials.json"
    write_json(path, [{"trial_id": "one"}])
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, [{"objective": float("nan")}])
    assert path.read_bytes() == original
    write_json(path, [{"trial_id": "one"}, {"trial_id": "two"}])
    assert len(json.loads(path.read_text())) == 2
    assert not path.with_suffix(".json.tmp").exists()


def test_csv_preserves_heterogeneous_fields_and_structured_candidates(tmp_path):
    rows = [{"trial": 1, "candidate": [1, 0]}, {"trial": 2, "error": "failed"}]
    path = tmp_path / "trials.csv"
    write_csv(path, rows, json_fields=("candidate",))
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        saved = list(reader)
        assert reader.fieldnames == ["trial", "candidate", "error"]
    assert json.loads(saved[0]["candidate"]) == [1, 0]
    assert saved[1]["error"] == "failed"
    assert rows[0]["candidate"] == [1, 0]


def test_csv_empty_exports_keep_explicit_schema(tmp_path):
    path = tmp_path / "summary.csv"
    write_csv(path, [])
    assert not path.exists()
    write_csv(path, [], fieldnames=("method", "objective"))
    assert path.read_text() == "method,objective\n"
