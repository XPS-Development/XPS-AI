"""
VAMAS 1988 subset parser for electron spectroscopy import.

Original implementation of the Surface Chemical Analysis Standard Data
Transfer Format (Dench et al., Surf. Interface Anal. 13, 1988). Supports
``NORM`` experiments with ``REGULAR`` scans and electron spectroscopy
techniques (XPS, UPS, AES, ELS, EDX, XRF). Not derived from third-party
GPL code.

We assume kinetic energy scales refer to the Fermi level, not the vacuum
level at the spectrometer.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core.metadata import SpectrumMetadata

from .types import ParsedSpectrum

_SUPPORTED_EXPERIMENT_MODES = frozenset({"NORM"})
_SUPPORTED_SCAN_MODES = frozenset({"REGULAR"})
_SUPPORTED_TECHNIQUES = frozenset(
    {
        "XPS",
        "UPS",
        "AES diff",
        "AES dir",
        "ELS",
        "EDX",
        "XRF",
    }
)
_DIFF_WIDTH_TECHNIQUES = frozenset({"AES diff"})


class UnsupportedVamasFormat(ValueError):
    """Raised when a VAMAS file is outside the supported subset."""


@dataclass(frozen=True)
class _VamasHeader:
    """Header fields required to parse blocks in the supported subset."""

    experiment_mode: str
    scan_mode: str
    experimental_variable_count: int
    num_future_upgrade_block_entries: int
    num_blocks: int


@dataclass(frozen=True)
class _VamasBlock:
    """Parsed spectral block with energy axis metadata and ordinate arrays."""

    name: str
    sample: str
    technique: str
    source_energy: float
    abscissa_label: str
    abscissa_start: float
    abscissa_increment: float
    num_corresponding_variables: int
    num_ordinate_values: int
    ordinates: NDArray[np.float64]


class _LineReader:
    """Sequential reader over stripped VAMAS text lines."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)

    def next_str(self) -> str:
        """Return the next non-empty line (stripped)."""
        try:
            return next(self._lines).strip()
        except StopIteration as exc:
            raise ValueError("Unexpected end of VAMAS file") from exc

    def next_int(self) -> int:
        """Return the next line as an integer."""
        return int(self.next_str())

    def next_float(self) -> float:
        """Return the next line as a float."""
        return float(self.next_str())

    def expect(self, text: str) -> None:
        """Consume a line and verify it equals ``text``."""
        line = self.next_str()
        if line != text:
            msg = f"Expected {text!r}, got {line!r}"
            raise ValueError(msg)


def _read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.read().splitlines()


def _parse_header(reader: _LineReader) -> _VamasHeader:
    _ = reader.next_str()  # format identifier
    _ = reader.next_str()  # institution
    _ = reader.next_str()  # instrument
    _ = reader.next_str()  # operator
    _ = reader.next_str()  # experiment title

    comment_count = reader.next_int()
    for _ in range(comment_count):
        _ = reader.next_str()

    experiment_mode = reader.next_str()
    scan_mode = reader.next_str()

    if experiment_mode not in _SUPPORTED_EXPERIMENT_MODES:
        msg = f"Unsupported VAMAS experiment_mode {experiment_mode!r}"
        raise UnsupportedVamasFormat(msg)
    if scan_mode not in _SUPPORTED_SCAN_MODES:
        msg = f"Unsupported VAMAS scan_mode {scan_mode!r}"
        raise UnsupportedVamasFormat(msg)

    _ = reader.next_int()  # num_spectral_regions for NORM

    experimental_variable_count = reader.next_int()
    for _ in range(experimental_variable_count):
        _ = reader.next_str()  # name
        _ = reader.next_str()  # units

    inclusion_count = reader.next_int()
    for _ in range(inclusion_count):
        _ = reader.next_str()

    manual_count = reader.next_int()
    for _ in range(manual_count):
        _ = reader.next_str()

    _ = reader.next_int()  # future upgrade experiment entry count
    num_future_upgrade_block_entries = reader.next_int()
    num_blocks = reader.next_int()

    return _VamasHeader(
        experiment_mode=experiment_mode,
        scan_mode=scan_mode,
        experimental_variable_count=experimental_variable_count,
        num_future_upgrade_block_entries=num_future_upgrade_block_entries,
        num_blocks=num_blocks,
    )


def _parse_block(header: _VamasHeader, reader: _LineReader) -> _VamasBlock:
    name = reader.next_str()
    sample = reader.next_str()

    for _ in range(7):
        _ = reader.next_int()  # date/time fields

    comment_count = reader.next_int()
    for _ in range(comment_count):
        _ = reader.next_str()

    technique = reader.next_str()
    if technique not in _SUPPORTED_TECHNIQUES:
        msg = (
            f"Unsupported VAMAS technique {technique!r} "
            f"(experiment_mode={header.experiment_mode!r}, "
            f"scan_mode={header.scan_mode!r})"
        )
        raise UnsupportedVamasFormat(msg)

    for _ in range(header.experimental_variable_count):
        _ = reader.next_float()

    _ = reader.next_str()  # analysis source
    source_energy = reader.next_float()
    _ = reader.next_float()  # source strength
    _ = reader.next_float()  # beam width x
    _ = reader.next_float()  # beam width y
    _ = reader.next_float()  # source polar angle
    _ = reader.next_float()  # source azimuth
    _ = reader.next_str()  # analyser mode
    _ = reader.next_float()  # analyser pass energy

    if technique in _DIFF_WIDTH_TECHNIQUES:
        _ = reader.next_float()

    _ = reader.next_float()  # analyser magnification
    _ = reader.next_float()  # work function / filter pass energy
    _ = reader.next_float()  # target bias
    _ = reader.next_float()  # analysis width x
    _ = reader.next_float()  # analysis width y
    _ = reader.next_float()  # analyser polar angle
    _ = reader.next_float()  # analyser azimuth
    _ = reader.next_str()  # species
    _ = reader.next_str()  # transition / charge state label
    _ = reader.next_int()  # charge of detected particle

    abscissa_label = reader.next_str()
    _ = reader.next_str()  # abscissa units
    abscissa_start = reader.next_float()
    abscissa_increment = reader.next_float()

    num_corresponding_variables = reader.next_int()
    for _ in range(num_corresponding_variables):
        _ = reader.next_str()  # label
        _ = reader.next_str()  # units

    _ = reader.next_str()  # signal mode
    _ = reader.next_float()  # signal collection time
    _ = reader.next_int()  # number of scans
    _ = reader.next_float()  # signal time correction

    _ = reader.next_float()  # sample normal tilt polar
    _ = reader.next_float()  # sample normal tilt azimuth
    _ = reader.next_float()  # sample rotation angle

    additional_count = reader.next_int()
    for _ in range(additional_count):
        _ = reader.next_str()
        _ = reader.next_str()
        _ = reader.next_float()

    for _ in range(header.num_future_upgrade_block_entries):
        _ = reader.next_str()

    num_ordinate_values = reader.next_int()
    for _ in range(num_corresponding_variables):
        _ = reader.next_float()  # minimum ordinate
        _ = reader.next_float()  # maximum ordinate

    ordinates = np.fromiter(
        (_parse_ordinate(reader) for _ in range(num_ordinate_values)),
        dtype=np.float64,
        count=num_ordinate_values,
    )

    return _VamasBlock(
        name=name,
        sample=sample,
        technique=technique,
        source_energy=source_energy,
        abscissa_label=abscissa_label,
        abscissa_start=abscissa_start,
        abscissa_increment=abscissa_increment,
        num_corresponding_variables=num_corresponding_variables,
        num_ordinate_values=num_ordinate_values,
        ordinates=ordinates,
    )


def _parse_ordinate(reader: _LineReader) -> float:
    return reader.next_float()


def _block_x_axis(
    block: _VamasBlock,
    *,
    use_binding_energy: bool,
) -> NDArray[np.float64] | None:
    """Build the requested energy axis from abscissa metadata."""
    n_pts = block.num_ordinate_values // block.num_corresponding_variables
    if n_pts <= 0:
        return None

    abscissa = np.linspace(
        block.abscissa_start,
        block.abscissa_start + (n_pts - 1) * block.abscissa_increment,
        n_pts,
        dtype=np.float64,
    )
    label = block.abscissa_label.lower()

    if "kinetic" in label:
        kinetic = abscissa
        binding = block.source_energy - kinetic
    elif "binding" in label:
        binding = abscissa
        kinetic = block.source_energy - binding
    else:
        return None

    return binding if use_binding_energy else kinetic


def _block_y_values(block: _VamasBlock) -> NDArray[np.float64]:
    n_vars = block.num_corresponding_variables
    n_pts = block.num_ordinate_values // n_vars
    data = block.ordinates.reshape(n_pts, n_vars).T
    return np.asarray(data[0], dtype=np.float64)


def _block_to_parsed_spectrum(
    block: _VamasBlock,
    path: Path,
    *,
    use_binding_energy: bool,
) -> ParsedSpectrum | None:
    x = _block_x_axis(block, use_binding_energy=use_binding_energy)
    if x is None:
        return None

    y = _block_y_values(block)
    if len(x) != len(y):
        return None

    return ParsedSpectrum(
        x=x,
        y=y,
        metadata=SpectrumMetadata(
            name=block.name,
            group=block.sample or "",
            file=str(path),
        ),
    )


def parse_vamas(
    path: str | Path,
    *,
    use_binding_energy: bool = True,
    use_cps: bool = True,
) -> list[ParsedSpectrum]:
    """
    Parse a VAMAS file and return :class:`ParsedSpectrum` instances for each block.

    Parameters
    ----------
    path : str or Path
        Path to the VAMAS (``.vms`` or ``.vamas``) file.
    use_binding_energy : bool, optional
        If True, use binding energy for the x-axis; otherwise kinetic energy.
    use_cps : bool, optional
        Ignored for VAMAS; the first corresponding variable (typically counts)
        is always used for intensities.

    Returns
    -------
    list[ParsedSpectrum]
        One spectrum per supported block in the file.

    Raises
    ------
    UnsupportedVamasFormat
        If the file uses an unsupported experiment mode, scan mode, or technique.
    ValueError
        If the file structure is invalid or truncated.
    """
    _ = use_cps
    file_path = Path(path)
    reader = _LineReader(_read_lines(file_path))
    header = _parse_header(reader)

    result: list[ParsedSpectrum] = []
    for _ in range(header.num_blocks):
        block = _parse_block(header, reader)
        spectrum = _block_to_parsed_spectrum(
            block,
            file_path,
            use_binding_energy=use_binding_energy,
        )
        if spectrum is not None:
            result.append(spectrum)

    reader.expect("end of experiment")
    return result
