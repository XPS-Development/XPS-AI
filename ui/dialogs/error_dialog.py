"""Dialog that shows a full exception report the user can copy."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


class ExceptionDialog(QDialog):
    """
    Modal dialog with a selectable, copyable exception report.

    Parameters
    ----------
    title : str
        Window title.
    details : str
        Full report text (type, message, traceback).
    parent : QWidget or None, optional
        Parent window.
    """

    def __init__(
        self,
        title: str,
        details: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 420)
        self._details = details

        heading = QLabel("An unexpected error occurred. You can copy the details below.", self)
        heading.setWordWrap(True)

        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)
        self._text.setPlainText(details)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        buttons = QDialogButtonBox(self)
        copy_btn = buttons.addButton("Copy", QDialogButtonBox.ButtonRole.ActionRole)
        close_btn = buttons.addButton(QDialogButtonBox.StandardButton.Close)
        copy_btn.clicked.connect(self._on_copy)
        close_btn.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(self._text, stretch=1)
        layout.addWidget(buttons)

    def _on_copy(self) -> None:
        """Copy the full report to the clipboard."""
        QApplication.clipboard().setText(self._details)


def show_exception_dialog(
    title: str,
    details: str,
    parent: QWidget | None = None,
) -> None:
    """
    Show :class:`ExceptionDialog` modally.

    Parameters
    ----------
    title : str
        Window title.
    details : str
        Full report text.
    parent : QWidget or None, optional
        Parent window; defaults to the active window when omitted.
    """
    app = QApplication.instance()
    if parent is None and isinstance(app, QApplication):
        parent = app.activeWindow()
    dialog = ExceptionDialog(title, details, parent=parent)
    dialog.exec()
