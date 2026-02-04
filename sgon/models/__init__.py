"""Model components for SGON."""

from .sgon import SGON1D
from .sgon_multiscale import SGON1DCoarseFine, SGON1DThreeScale

__all__ = ["SGON1D", "SGON1DCoarseFine", "SGON1DThreeScale"]
