"""
Import service: parse spectrum files and produce Change objects.

Produces CompositeChange with CreateSpectrum and SetSpectrumMetadata
for each spectrum in the file.
"""

from uuid import uuid4
from pathlib import Path

from tools.parsers import parse_spectrum_file

from .command.changes import CompositeChange, CreateSpectrum, SetMetadata


def import_spectra(
    path: str | Path,
    *,
    use_binding_energy: bool = True,
    use_cps: bool = True,
) -> CompositeChange:
    """
    Parse a spectrum file and return a CompositeChange to create spectra with metadata.

    Parameters
    ----------
    path : str or Path
        Path to the spectrum file (.txt, .dat, .vms, .vamas).
    use_binding_energy : bool, optional
        Whether to use binding energy as the x-axis.
    use_cps : bool, optional
        Whether to use CPS as the y-axis.
    Returns
    -------
    CompositeChange
        Change containing CreateSpectrum and SetMetadata for each spectrum.
    """
    parsed = parse_spectrum_file(path, use_binding_energy=use_binding_energy, use_cps=use_cps)
    changes = []

    for i, ps in enumerate(parsed):
        sid = f"s{uuid4().hex}"
        changes.append(CreateSpectrum(x=ps.x, y=ps.y, spectrum_id=sid))
        changes.append(SetMetadata(obj_id=sid, metadata=ps.metadata))

    return CompositeChange(changes=changes)
