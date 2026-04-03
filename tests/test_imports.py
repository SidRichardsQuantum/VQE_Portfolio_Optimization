import importlib


def test_base_import():
    importlib.import_module("vqe_portfolio")


def test_core_api_imports():
    m = importlib.import_module("vqe_portfolio")

    # Algorithms
    assert hasattr(m, "run_fractional_vqe")
    assert hasattr(m, "run_binary_vqe")
    assert hasattr(m, "run_qaoa")

    # Configs
    assert hasattr(m, "BinaryVQEConfig")
    assert hasattr(m, "FractionalVQEConfig")
    assert hasattr(m, "QAOAConfig")

    # Results
    assert hasattr(m, "BinaryVQEResult")
    assert hasattr(m, "FractionalVQEResult")
    assert hasattr(m, "QAOAResult")

    # sweeps
    assert hasattr(m, "binary_lambda_sweep")
    assert hasattr(m, "fractional_lambda_sweep")
    assert hasattr(m, "qaoa_lambda_sweep")

    # utilities
    assert hasattr(m, "set_global_seed")
