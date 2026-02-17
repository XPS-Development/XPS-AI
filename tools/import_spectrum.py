"""
Unified spectrum file parsing and format dispatch.

Provides ParsedSpectrum and parse_spectrum_file() to extract spectra from
.txt (casa-like), .dat (two-column), and VAMAS formats.
"""

from pathlib import Path

from tools.parsers.casa import parse_casa_txt
from tools.parsers.dat import parse_dat
from tools.parsers.types import ParsedSpectrum
from tools.parsers.vamas import parse_vamas

__all__ = ["ParsedSpectrum", "parse_spectrum_file"]


def parse_spectrum_file(path: str | Path) -> list[ParsedSpectrum]:
    """
    Parse a spectrum file and return extracted spectra.

    Dispatches by file extension:
    - .txt -> casa-like format
    - .dat -> two-column x, y format
    - .vms, .vamas -> VAMAS format

    Parameters
    ----------
    path : str or Path
        Path to the spectrum file.

    Returns
    -------
    list[ParsedSpectrum]
        One spectrum for .txt/.dat, multiple for VAMAS.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    path = Path(path) if isinstance(path, str) else path
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return parse_casa_txt(path)
    if suffix == ".dat":
        return parse_dat(path)
    if suffix in (".vms", ".vamas"):
        return parse_vamas(path)

    raise ValueError(f"Unsupported file extension: {suffix}")
