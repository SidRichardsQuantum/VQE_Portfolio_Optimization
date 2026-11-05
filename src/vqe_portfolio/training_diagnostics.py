"""Compatibility alias for :mod:`vqe_portfolio.research.training_diagnostics`."""

import sys

from .research import training_diagnostics as _implementation

sys.modules[__name__] = _implementation
