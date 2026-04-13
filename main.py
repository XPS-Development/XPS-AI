import sys
import traceback

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication, QMessageBox

from app.error_dump import enable_user_exception_ui, orchestrator_error_user_feedback_done, save_error_dump
from ui.controller import ControllerWrapper
from ui.main_window import MainWindow


class _SafeNotifyApplication(QApplication):
    """
    QApplication that catches Python exceptions during event delivery.

    Qt slots and other callbacks invoked through ``notify`` are wrapped so
    failures are written to ``error_dumps`` and shown in a message box instead
    of failing silently or terminating the process without feedback.
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
            if orchestrator_error_user_feedback_done(exc):
                return False
            dump_path = save_error_dump(exc)
            message = f"{exc}\n\nDetails were saved to:\n{dump_path}"
            parent = self.activeWindow()
            QMessageBox.critical(parent, "Unexpected error", message)
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
    enable_user_exception_ui()

    window: MainWindow | None = None
    try:
        controller = ControllerWrapper()
        window = MainWindow(controller)
        window.show()
        return app.exec()
    except Exception as exc:
        dump_path = save_error_dump(exc)
        message = (
            "An unexpected error occurred and the application needs to close.\n\n"
            f"Details were saved to:\n{dump_path}"
        )
        if window is not None:
            QMessageBox.critical(window, "Unexpected error", message)
        else:
            # Fallback if the window was not created yet.
            sys.stderr.write(message + "\n")
            traceback.print_exception(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
