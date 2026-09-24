"""Application entry point: start the Qt main window."""

import sys

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication

from app.error_dump import (
    enable_user_exception_ui,
    orchestrator_error_user_feedback_done,
    report_exception,
)
from ui.assets import APP_NAME, load_app_icon
from ui.controller import ControllerWrapper
from ui.main_window import MainWindow
from ui.theme import install_theme


class _SafeNotifyApplication(QApplication):
    """
    QApplication that catches Python exceptions during event delivery.

    Qt slots and other callbacks invoked through ``notify`` are wrapped so
    failures are shown in a copyable dialog instead of failing silently or
    terminating the process without feedback.
    """

    def notify(self, receiver: QObject, event: QEvent) -> bool:
        """
        Deliver an event to ``receiver``, handling unexpected Python exceptions.

        Parameters
        ----------
        receiver
            Target object for the event.
        event
            Qt event instance.

        Returns
        -------
        bool
            Event filter / delivery result from the base implementation, or
            ``False`` if a Python exception was handled here.
        """
        try:
            return super().notify(receiver, event)
        except Exception as exc:
            if not orchestrator_error_user_feedback_done(exc):
                report_exception(exc, title="Unexpected error")
            return False


def main() -> int:
    """
    Application entry point for the Qt UI.

    Creates the Qt application, controller wrapper, and main window, then
    starts the event loop.

    Returns
    -------
    int
        Exit code from the Qt event loop.
    """
    app = _SafeNotifyApplication(sys.argv)
    install_theme(app)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    if getattr(sys, "frozen", False):
        app.setWindowIcon(load_app_icon())
    enable_user_exception_ui()

    try:
        controller = ControllerWrapper()
        window = MainWindow(controller)
        window.show()
        return app.exec()
    except Exception as exc:
        report_exception(
            exc,
            title="Unexpected error",
            context="application startup",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
