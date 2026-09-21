"""
Parser for casa-like .txt spectrum files.

Format: Line 1 = spectrum name, Line 2 = header, Line 3 = column headers
(K.E., Counts, B.E., CPS), Lines 4+ = tab-separated data.
"""

from pathlib import Path

import numpy as np

from core.metadata import SpectrumMetadata

from .types import ParsedSpectrum


def parse_casa_txt(
    path: Path,
    *,
    use_binding_energy: bool = True,
    use_cps: bool = True,
) -> list[ParsedSpectrum]:
    """
    Parse a casa-like .txt spectrum file.

    Parameters
    ----------
    path : Path
        Path to the .txt file.
    use_binding_energy : bool, optional
        If True, use B.E. column for x; otherwise use K.E. Default True.
    use_cps : bool, optional
        If True, use CPS column for y; otherwise use Counts. Default True.

    Returns
    -------
    list[ParsedSpectrum]
        Single-element list with the parsed spectrum.

    Raises
    ------
    ValueError
        If the file has fewer than 4 lines.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise ValueError(f"Casa .txt file has too few lines: {path}")

    name = lines[0].strip()
    # Line 2: header (Characteristic Energy, Acquisition Time)
    # Line 3: K.E.  Counts    B.E.  CPS
    # Data: columns 0=K.E., 1=Counts, 2=empty, 3=B.E., 4=CPS
    data = np.loadtxt(
        lines[3:],
        delimiter="\t",
        usecols=(0, 1, 3, 4),
        dtype=np.float64,
    )

    if use_binding_energy:
        x = data[:, 2]
    else:
        x = data[:, 0]
    if use_cps:
        y = data[:, 3]
    else:
        y = data[:, 1]

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    return [
        ParsedSpectrum(
            x=x,
            y=y,
            metadata=SpectrumMetadata(
                name=name,
                group="",
                file=str(path),
            ),
        )
    ]
