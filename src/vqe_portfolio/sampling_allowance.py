"""Compatibility alias for :mod:`vqe_portfolio.research.sampling_allowance`."""

import sys

from .research import sampling_allowance as _implementation

sys.modules[__name__] = _implementation
