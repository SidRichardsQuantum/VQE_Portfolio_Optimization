import importlib
import pytest


def _module_available(name: str) -> bool:
    spec = importlib.util.find_spec(name)
    return spec is not None


@pytest.mark.skipif(not _module_available("cvxpy"), reason="cvxpy not installed (markowitz extra not enabled)")
def test_import_cvxpy():
    import cvxpy  # noqa: F401


@pytest.mark.skipif(not _module_available("osqp"), reason="osqp not installed (markowitz extra not enabled)")
def test_import_osqp():
    import osqp  # noqa: F401
