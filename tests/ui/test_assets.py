"""Tests for bundled asset path helpers."""

from __future__ import annotations

import sys

from ui.assets import bundle_root, icon_path


def test_icon_path_points_at_repo_assets() -> None:
    """Development runs resolve icons from the repository assets folder."""
    path = icon_path("app.png")
    assert path.is_file()
    assert path.name == "app.png"
    assert path.parent == bundle_root() / "assets" / "icons"


def test_windows_ico_is_bundled() -> None:
    """A multi-size ICO is available for the Windows executable and installer."""
    path = icon_path("app.ico")
    assert path.is_file()
    assert path.read_bytes()[:4] == b"\x00\x00\x01\x00"


def test_bundle_root_uses_meipass_when_frozen(tmp_path, monkeypatch) -> None:
    """PyInstaller builds read assets from ``sys._MEIPASS``."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert bundle_root() == tmp_path
    assert icon_path("app.png") == tmp_path / "assets" / "icons" / "app.png"
