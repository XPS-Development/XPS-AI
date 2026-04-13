"""
Parser for two-column .dat spectrum files (x, y).
"""

from pathlib import Path

import numpy as np

from core.metadata import SpectrumMetadata

from .types import ParsedSpectrum


def parse_dat(path: Path) -> list[ParsedSpectrum]:
    """
    Parse a two-column .dat spectrum file (whitespace-separated x, y).

    Parameters
    ----------
    path : Path
        Path to the .dat file.

    Returns
    -------
    list[ParsedSpectrum]
        Single-element list with the parsed spectrum.

    Raises
    ------
    ValueError
        If the file has fewer than two columns or an odd number of values.
    """
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        if data.size < 2:
            raise ValueError(f"Expected at least two values in .dat file: {path}")
        try:
            data = data.reshape(-1, 2)
        except ValueError:
            raise ValueError(f"Expected at least two columns in .dat file: {path}") from None
    elif data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in .dat file: {path}")

    x = np.asarray(data[:, 0], dtype=np.float64)
    y = np.asarray(data[:, 1], dtype=np.float64)

    stem = path.stem
    return [
        ParsedSpectrum(
            x=x,
            y=y,
            metadata=SpectrumMetadata(
                name=stem,
                group="",
                file=str(path),
            ),
        )
    ]
