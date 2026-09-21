"""Resolve paths to bundled UI assets."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon

APP_NAME = "XPS-AI"


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
