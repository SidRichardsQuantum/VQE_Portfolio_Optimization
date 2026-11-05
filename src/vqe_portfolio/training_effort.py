"""Compatibility alias for :mod:`vqe_portfolio.research.training_effort`."""

import sys

from .research import training_effort as _implementation

sys.modules[__name__] = _implementation
