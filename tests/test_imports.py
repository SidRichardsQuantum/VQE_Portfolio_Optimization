import importlib


def test_base_import():
    importlib.import_module("vqe_portfolio")


def test_core_api_imports():
    m = importlib.import_module("vqe_portfolio")
    # Touch attributes so linting and the test both ensure they exist.
    assert hasattr(m, "run_fractional_vqe")
    assert hasattr(m, "run_binary_vqe")
    assert hasattr(m, "BinaryVQEConfig")
    assert hasattr(m, "FractionalVQEConfig")
