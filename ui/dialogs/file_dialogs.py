"""Helpers for Qt file dialogs."""

from __future__ import annotations

import re
from pathlib import Path

_FILTER_EXT_RE = re.compile(r"\*\.([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)")

SPECTRUM_OPEN_SUFFIXES = frozenset({".txt", ".csv", ".dat", ".vms", ".vamas"})


def ensure_suffix_from_filter(
    path: str | Path,
    selected_filter: str,
    *,
    fallback: str | None = None,
) -> Path:
    """
    Append a file extension from the selected save-dialog filter when missing.

    Native save dialogs on Linux often return the typed name without applying
    the filter's ``*.ext`` pattern; Windows and macOS native dialogs usually do.

    Parameters
    ----------
    path : str or Path
        Path returned by ``QFileDialog.getSaveFileName``.
    selected_filter : str
        Filter chosen by the user (e.g. ``"JSON files (*.json)"``).
    fallback : str or None, optional
        Extension (with or without a leading dot) when the filter has no
        ``*.ext`` pattern, such as ``"All files (*)"``.

    Returns
    -------
    Path
        ``path`` unchanged if it already has a suffix; otherwise with an
        extension from ``selected_filter`` or ``fallback``.
    """
    result = Path(path)
    if result.suffix:
        return result
    match = _FILTER_EXT_RE.search(selected_filter)
    if match is not None:
        return result.with_suffix(f".{match.group(1).lower()}")
    if fallback:
        suffix = fallback if fallback.startswith(".") else f".{fallback}"
        return result.with_suffix(suffix)
    return result


def split_open_paths(paths: list[str] | list[Path]) -> tuple[list[Path], list[Path]]:
    """
    Split Open-dialog selections into spectrum imports vs collection files.

    Anything that is not a known spectrum suffix is treated as a collection
    candidate. That covers ``.json``, gzip (``.json.gz`` / ``.gz``), and bare
    names produced by Linux save dialogs that omit the filter extension.

    Parameters
    ----------
    paths : list of str or Path
        Paths chosen in the Open dialog.

    Returns
    -------
    tuple of (list of Path, list of Path)
        ``(spectrum_paths, collection_paths)`` in the original order.
    """
    spectrum_paths: list[Path] = []
    collection_paths: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.suffix.lower() in SPECTRUM_OPEN_SUFFIXES:
            spectrum_paths.append(path)
        else:
            collection_paths.append(path)
    return spectrum_paths, collection_paths
