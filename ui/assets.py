"""Resolve paths to bundled UI assets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QIcon

from app.paths import bundle_root

if TYPE_CHECKING:
    from pathlib import Path

APP_NAME = "XPS-AI"


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
    return bundle_root() / "assets" / "icons" / name


def load_app_icon() -> QIcon:
    """
    Return the application icon, preferring the multi-size ``.ico``.

    Returns
    -------
    QIcon
        Window / taskbar icon loaded from bundled assets.
    """
    icon = QIcon()
    for name in ("app.ico", "app.png"):
        path = icon_path(name)
        if path.is_file():
            icon.addFile(str(path))
    return icon
