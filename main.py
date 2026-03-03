from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

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

    controller = ControllerWrapper()
    window = MainWindow(controller)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

