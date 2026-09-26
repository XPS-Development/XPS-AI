"""Tests for bundled resource paths."""

from __future__ import annotations

import sys

from app.paths import bundle_root, resolve_resource_path


def test_resolve_resource_path_keeps_absolute_paths() -> None:
    """An absolute model path is passed through unchanged."""
    assert resolve_resource_path("/some/model.onnx") == "/some/model.onnx"


def test_resolve_resource_path_uses_bundle_root() -> None:
    """A relative model path is resolved from the repository or frozen bundle."""
    resolved = resolve_resource_path("assets/models/model.onnx")
    assert resolved == str(bundle_root() / "assets" / "models" / "model.onnx")


def test_resolve_resource_path_uses_meipass_when_frozen(tmp_path, monkeypatch) -> None:
    """PyInstaller builds resolve relative paths from ``sys._MEIPASS``."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    resolved = resolve_resource_path("assets/models/model.onnx")
    assert resolved == str(tmp_path / "assets" / "models" / "model.onnx")


def test_resolve_resource_path_empty_is_none() -> None:
    """Blank settings mean no model path."""
    assert resolve_resource_path(None) is None
    assert resolve_resource_path("  ") is None
