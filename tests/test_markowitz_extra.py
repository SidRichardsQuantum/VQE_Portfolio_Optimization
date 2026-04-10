from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def _load_pyproject() -> dict:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        return tomllib.load(f)


def test_markowitz_extra_declares_cvxpy_dependency():
    data = _load_pyproject()
    deps = data["project"]["optional-dependencies"]["markowitz"]

    assert "cvxpy>=1.4,<2.0" in deps


def test_markowitz_extra_declares_osqp_dependency():
    data = _load_pyproject()
    deps = data["project"]["optional-dependencies"]["markowitz"]

    assert "osqp>=0.6,<1.0" in deps
