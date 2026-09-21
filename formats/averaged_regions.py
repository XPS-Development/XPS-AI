"""
Parser for wide averaged-region tables (``.csv`` / headered ``.dat``).

Each pair of columns ``{line}_BE_eV`` / ``{line}_Intensity_cps`` becomes one
spectrum. Columns may have different lengths (trailing empty cells).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from core.metadata import SpectrumMetadata

from .types import ParsedSpectrum

_INTENSITY_SUFFIX = "_intensity_cps"
_HEADER_PAIR_RE = re.compile(
    r"^(.+)_be_ev$",
    re.IGNORECASE,
)


def looks_like_averaged_regions_header(header_line: str) -> bool:
    """
    Return True if ``header_line`` looks like an averaged-regions wide header.

    Parameters
    ----------
    header_line : str
        First line of a candidate file (not yet split).

    Returns
    -------
    bool
        True when at least one ``{name}_BE_eV`` / ``{name}_Intensity_cps`` pair
        is present.
    """
    delimiter = _guess_delimiter(header_line)
    columns = next(csv.reader([header_line], delimiter=delimiter))
    return bool(_column_pairs(columns))


def parse_averaged_regions(path: Path) -> list[ParsedSpectrum]:
    """
    Parse a wide averaged-regions table into one spectrum per line pair.

    Parameters
    ----------
    path : Path
        Path to the ``.csv`` or headered ``.dat`` file.

    Returns
    -------
    list[ParsedSpectrum]
        One entry per ``{line}_BE_eV`` / ``{line}_Intensity_cps`` column pair.

    Raises
    ------
    ValueError
        If the file is empty, has no matching column pairs, or yields no points.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"Averaged-regions file is empty: {path}")

    delimiter = _guess_delimiter(lines[0], suffix=path.suffix.lower())
    reader = csv.reader(lines, delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError(f"Averaged-regions file is empty: {path}") from None

    pairs = _column_pairs(header)
    if not pairs:
        raise ValueError(f"No {{line}}_BE_eV / {{line}}_Intensity_cps column pairs in: {path}")

    rows = list(reader)
    results: list[ParsedSpectrum] = []
    for name, be_idx, intensity_idx in pairs:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            x_cell = row[be_idx].strip() if be_idx < len(row) else ""
            y_cell = row[intensity_idx].strip() if intensity_idx < len(row) else ""
            if not x_cell or not y_cell:
                continue
            try:
                xs.append(float(x_cell))
                ys.append(float(y_cell))
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric value in averaged-regions file {path}: {x_cell!r}, {y_cell!r}"
                ) from exc

        if not xs:
            raise ValueError(f"No data points for spectrum {name!r} in: {path}")

        results.append(
            ParsedSpectrum(
                x=np.asarray(xs, dtype=np.float64),
                y=np.asarray(ys, dtype=np.float64),
                metadata=SpectrumMetadata(
                    name=name,
                    group="",
                    file=str(path),
                ),
            )
        )

    return results


def _guess_delimiter(header_line: str, *, suffix: str = "") -> str:
    """Choose field delimiter from suffix and header content."""
    if suffix == ".csv":
        return ","
    if suffix == ".dat":
        return "\t" if "\t" in header_line else ","
    if "\t" in header_line:
        return "\t"
    if "," in header_line:
        return ","
    try:
        dialect = csv.Sniffer().sniff(header_line, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        return ","


def _column_pairs(header: list[str]) -> list[tuple[str, int, int]]:
    """
    Map header columns to ``(name, be_index, intensity_index)`` triples.

    Returns
    -------
    list[tuple[str, int, int]]
        Ordered by appearance of the BE column.
    """
    lower_to_index: dict[str, int] = {}
    for i, col in enumerate(header):
        key = col.strip().lower()
        if key and key not in lower_to_index:
            lower_to_index[key] = i

    pairs: list[tuple[str, int, int]] = []
    seen_names: set[str] = set()
    for i, col in enumerate(header):
        stripped = col.strip()
        match = _HEADER_PAIR_RE.match(stripped)
        if match is None:
            continue
        name = match.group(1)
        intensity_key = f"{name.lower()}{_INTENSITY_SUFFIX}"
        intensity_idx = lower_to_index.get(intensity_key)
        if intensity_idx is None:
            continue
        name_key = name.lower()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        pairs.append((name, i, intensity_idx))

    return pairs
