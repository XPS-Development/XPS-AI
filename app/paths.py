"""Paths that stay valid in a source checkout and in a PyInstaller bundle."""

from __future__ import annotations

import sys
from pathlib import Path


def bundle_root() -> Path:
    """
    Return the application bundle root.

    In a PyInstaller build this is ``sys._MEIPASS`` (the folder next to the
    executable when ``contents_directory='.'``). During development it is the
    repository root.

    Returns
    -------
    Path
        Directory that contains ``assets/``.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resolve_resource_path(path: str | Path | None) -> str | None:
    """
    Resolve a bundled resource path.

    Absolute paths are returned unchanged. Relative paths are resolved from
    :func:`bundle_root`, so ``assets/models/model.onnx`` finds the file next
    to the frozen executable rather than in the process working directory.

    Parameters
    ----------
    path : str or Path or None
        Resource path from settings, or ``None`` when unset.

    Returns
    -------
    str or None
        Absolute path string, or ``None`` when ``path`` is empty.
    """
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return str(candidate)
    return str(bundle_root() / candidate)
