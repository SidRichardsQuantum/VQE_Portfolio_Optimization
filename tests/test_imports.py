def test_base_import():
    import vqe_portfolio


def test_core_api_imports():
    from vqe_portfolio import (
        run_fractional_vqe,
        run_binary_vqe,
        BinaryVQEConfig,
        FractionalVQEConfig,
    )
