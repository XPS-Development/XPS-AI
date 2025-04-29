import sys

from PySide6.QtWidgets import *
from PySide6.QtGui import QPalette, QColor

from app.main_window import MainWindow


def main(light=False):
    app = QApplication(sys.argv)

    # Set the application palette to a light theme
    if light:
        set_light_palette(app)

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


def set_light_palette(app):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))  # Light gray background
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))  # Black text
    palette.setColor(QPalette.Base, QColor(255, 255, 255))  # White base
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))  # Light gray alternate base
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))  # Light yellow tooltip base
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))  # Black tooltip text
    palette.setColor(QPalette.Text, QColor(0, 0, 0))  # Black text
    palette.setColor(QPalette.Button, QColor(240, 240, 240))  # Light gray button
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))  # Black button text
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))  # White bright text
    palette.setColor(QPalette.Link, QColor(0, 0, 255))  # Blue link
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))  # Light blue highlight
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))  # White highlighted text

    app.setPalette(palette)


if __name__ == "__main__":
    main()
