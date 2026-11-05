"""Compatibility alias for :mod:`vqe_portfolio.research.failure_diagnostics`."""

import sys

from .research import failure_diagnostics as _implementation

sys.modules[__name__] = _implementation
