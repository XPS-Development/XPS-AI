"""Spectrum file parsers for casa-like .txt, .dat, and VAMAS formats."""

from .casa import parse_casa_txt
from .dat import parse_dat
from .vamas import parse_vamas

__all__ = ["parse_casa_txt", "parse_dat", "parse_vamas"]
