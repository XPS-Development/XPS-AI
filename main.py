import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from app.error_dump import save_error_dump
from ui.controller import ControllerWrapper
from ui.main_window import MainWindow


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
    app = QApplication(sys.argv)

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
