"""Resolve paths to bundled UI assets."""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ICONS_DIR = _PROJECT_ROOT / "assets" / "icons"


def icon_path(name: str) -> Path:
    r"""
    Return the filesystem path for a named icon file.

    Parameters
    ----------
    name : str
        Icon filename (e.g. ``\"plus.svg\"``).

    Returns
    -------
    Path
        Absolute path under ``assets/icons``.
    """
    return _ICONS_DIR / name
