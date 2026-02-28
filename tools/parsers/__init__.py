"""Spectrum file parsers for casa-like .txt, .dat, and VAMAS formats."""

from pathlib import Path

from .casa import parse_casa_txt
from .dat import parse_dat
from .vamas import parse_vamas
from .types import ParsedSpectrum

__all__ = ["ParsedSpectrum", "parse_spectrum_file", "parse_casa_txt", "parse_dat", "parse_vamas"]


def parse_spectrum_file(
    path: str | Path, *, use_binding_energy: bool = True, use_cps: bool = True
) -> list[ParsedSpectrum]:
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
    use_binding_energy : bool, optional
        If True, use binding energy axis for x; otherwise use kinetic energy.
        Default True.
    use_cps : bool, optional
        If True, use CPS axis for y; otherwise use counts.
        Default True.

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
        return parse_casa_txt(path, use_binding_energy=use_binding_energy, use_cps=use_cps)
    if suffix == ".dat":
        return parse_dat(path)
    if suffix in (".vms", ".vamas"):
        return parse_vamas(path, use_binding_energy=use_binding_energy, use_cps=use_cps)

    raise ValueError(f"Unsupported file extension: {suffix}")
