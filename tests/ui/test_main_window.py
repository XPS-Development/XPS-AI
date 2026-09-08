"""Tests for main-window document safety helpers (dirty title, close prompt)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox
from tests.conftest import seed_hierarchy_metadata

from ui.controller import ControllerWrapper
from ui.main_window import MainWindow

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast(QApplication, app)


@pytest.fixture
def window(
    qapp: QApplication, hierarchy_collection, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MainWindow]:
    """Build a MainWindow with hierarchy data and an unsaved rename."""
    del qapp
    ctrl = ControllerWrapper(collection=hierarchy_collection)
    seed_hierarchy_metadata(ctrl.orchestrator.ctx.metadata)
    ctrl.rename_spectrum("s1", "renamed")
    win = MainWindow(ctrl)
    yield win
    # Bypass the unsaved-changes prompt during fixture teardown.
    monkeypatch.setattr(win, "_confirm_close", lambda: True)
    win.close()


def test_window_title_shows_dirty_marker_for_untitled(window: MainWindow) -> None:
    """A never-saved dirty document shows ``*Untitled`` in the title."""
    assert window._controller.is_dirty is True
    window._update_window_title()
    assert window.windowTitle() == "Spectrum Viewer - *Untitled"


def test_window_title_clears_dirty_marker_after_save(window: MainWindow, tmp_path) -> None:
    """Saving clears the dirty marker from the window title."""
    path = tmp_path / "demo.json"
    window._controller.dump_collection(path)
    window._update_window_title()
    assert window._controller.is_dirty is False
    assert window.windowTitle() == "Spectrum Viewer - demo.json"


def test_confirm_close_accepts_when_clean(
    window: MainWindow, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clean documents close without prompting."""
    window._controller.dump_collection(tmp_path / "clean.json")
    prompted = {"called": False}

    def _boom(*_a, **_k):
        prompted["called"] = True
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", _boom)
    assert window._confirm_close() is True
    assert prompted["called"] is False


def test_confirm_close_cancel_keeps_window_open(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel on the unsaved prompt must refuse close."""
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_a, **_k: QMessageBox.StandardButton.Cancel,
    )
    assert window._controller.is_dirty is True
    assert window._confirm_close() is False


def test_confirm_close_discard_allows_close(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discard on the unsaved prompt must allow close."""
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_a, **_k: QMessageBox.StandardButton.Discard,
    )
    assert window._confirm_close() is True


def test_confirm_close_save_uses_try_save(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save on the unsaved prompt delegates to ``_try_save``."""
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_a, **_k: QMessageBox.StandardButton.Save,
    )
    monkeypatch.setattr(window, "_try_save", lambda: True)
    assert window._confirm_close() is True


def test_close_event_ignored_when_user_cancels(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``closeEvent`` ignores the event when the user cancels."""
    monkeypatch.setattr(window, "_confirm_close", lambda: False)
    event = MagicMock(spec=QCloseEvent)
    window.closeEvent(event)
    event.ignore.assert_called_once()
    event.accept.assert_not_called()


def test_close_event_accepted_when_confirmed(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``closeEvent`` accepts the event when close is confirmed."""
    monkeypatch.setattr(window, "_confirm_close", lambda: True)
    event = MagicMock(spec=QCloseEvent)
    window.closeEvent(event)
    event.accept.assert_called_once()
    event.ignore.assert_not_called()
