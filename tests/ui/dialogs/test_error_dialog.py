"""Tests for the copyable exception dialog."""

from __future__ import annotations

import sys
from typing import cast

import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QPlainTextEdit

from ui.dialogs.error_dialog import ExceptionDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast(QApplication, app)


def test_exception_dialog_shows_details_and_copies(qapp: QApplication) -> None:
    """Dialog displays the report and Copy puts it on the clipboard."""
    details = "Exception type: ValueError\nException message: boom\n"
    dialog = ExceptionDialog("Unexpected error", details)

    assert dialog.windowTitle() == "Unexpected error"
    text = dialog.findChild(QPlainTextEdit)
    assert text is not None
    assert details in text.toPlainText()

    box = dialog.findChild(QDialogButtonBox)
    assert box is not None
    copy_btn = next(b for b in box.buttons() if b.text() == "Copy")
    copy_btn.click()
    assert qapp.clipboard().text() == details
