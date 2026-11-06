"""Shared file exports and provenance for repository experiment runners."""

import csv
import hashlib
import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def write_json(path: Path, value) -> None:
    """Atomically replace a JSON checkpoint, rejecting non-finite numbers."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows, *, fieldnames=None, json_fields=()) -> None:
    """Export heterogeneous rows in stable field order without changing them.

    Explicit fieldnames also support header-only exports. Selected structured
    fields can be encoded as JSON rather than Python representations.
    """
    if not rows and fieldnames is None:
        return
    fields = (
        list(fieldnames)
        if fieldnames is not None
        else list(dict.fromkeys(key for row in rows for key in row))
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            for field in json_fields:
                if field in row:
                    row[field] = json.dumps(row[field], allow_nan=False)
            writer.writerow(row)


def provenance(root: Path, *runner_sources: Path, extra_packages=()) -> dict:
    """Record environment and recursive package hashes plus explicit runners.

    Paths are relative to the repository root so manifests remain portable.
    The caller supplies its root and entry point: installed package locations
    never determine experiment output paths. Historical manifests are untouched.
    """
    root = root.resolve()
    packages = {}
    for name in (
        "vqe-portfolio",
        "numpy",
        "pennylane",
        "scipy",
        "cvxpy",
        "osqp",
        "pyscipopt",
        "pennylane-lightning",
        *extra_packages,
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    sources = set((root / "src/vqe_portfolio").rglob("*.py"))
    sources.update(path.resolve() for path in runner_sources)
    cpu_model = platform.processor()
    if not cpu_model and Path("/proc/cpuinfo").exists():
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "",
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_model,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "packages": packages,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources)
        },
    }
