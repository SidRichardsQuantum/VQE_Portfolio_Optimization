"""Compatibility alias for :mod:`vqe_portfolio.baselines.solvers`."""

import sys

from .baselines import solvers as _implementation

sys.modules[__name__] = _implementation
