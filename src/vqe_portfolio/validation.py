"""Compatibility alias for :mod:`vqe_portfolio.research.validation`."""

import sys

from .research import validation as _implementation

sys.modules[__name__] = _implementation
