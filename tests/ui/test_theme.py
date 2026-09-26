"""Tests for :mod:`ui.theme` Fusion light theme and translucent popups."""

from __future__ import annotations

import sys
from typing import cast

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QMenu, QWidget

from ui.theme import (
    CHECKBOX_SIZE,
    EditorStyle,
    install_theme,
    make_translucent_popup,
    paint_rounded_checkbox,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast(QApplication, app)


def test_editor_style_makes_menus_translucent(qapp: QApplication) -> None:
    """Polish sets frameless, shadowless translucent flags on menus."""
    del qapp
    style = EditorStyle()
    menu = QMenu()
    style.polish(menu)
    flags = menu.windowFlags()
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.NoDropShadowWindowHint
    assert menu.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    style.polish(menu)
    assert menu.windowFlags() & Qt.WindowType.FramelessWindowHint
    menu.close()


def test_editor_style_ignores_regular_widgets(qapp: QApplication) -> None:
    """Ordinary widgets are not made translucent popups."""
    del qapp
    style = EditorStyle()
    widget = QWidget()
    style.polish(widget)
    assert not (widget.windowFlags() & Qt.WindowType.FramelessWindowHint)
    widget.close()


def test_opaque_popups_env_disables_translucency(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``XPS_AI_OPAQUE_POPUPS`` skips translucent menu chrome."""
    del qapp
    monkeypatch.setenv("XPS_AI_OPAQUE_POPUPS", "1")
    style = EditorStyle()
    menu = QMenu()
    style.polish(menu)
    assert not menu.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    menu.close()


def test_make_translucent_popup_is_idempotent(qapp: QApplication) -> None:
    """Repeated calls keep the same flags without error."""
    del qapp
    menu = QMenu()
    make_translucent_popup(menu)
    make_translucent_popup(menu)
    assert menu.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert menu.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    menu.close()


def test_install_theme_sets_fusion_light_and_rounds_menus(qapp: QApplication) -> None:
    """``install_theme`` installs Fusion, light theme, stylesheet, and menu polish."""
    app = qapp
    prev_style_name = app.style().name()
    try:
        style = install_theme(app)
        assert style.baseStyle().name().lower() == "fusion"
        assert app.palette().color(QPalette.ColorRole.Window).lightness() > 128
        assert app.styleSheet() != ""
        assert "QMessageBox" in app.styleSheet()
        assert "QCheckBox::indicator" in app.styleSheet()

        menu = QMenu()
        submenu = menu.addMenu("x")
        menu.ensurePolished()
        submenu.ensurePolished()
        assert menu.windowFlags() & Qt.WindowType.FramelessWindowHint
        assert submenu.windowFlags() & Qt.WindowType.FramelessWindowHint
        menu.close()
    finally:
        app.setStyleSheet("")
        app.setStyle(prev_style_name or "Fusion")
        app.setPalette(app.style().standardPalette())
        app.styleHints().unsetColorScheme()


def test_editor_style_checkbox_metrics_and_paint(qapp: QApplication) -> None:
    """Rounded checkbox indicators use a fixed size and paint without error."""
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QStyle, QStyleOptionButton

    del qapp
    style = EditorStyle()
    assert style.pixelMetric(QStyle.PixelMetric.PM_IndicatorWidth) == CHECKBOX_SIZE
    assert style.pixelMetric(QStyle.PixelMetric.PM_IndicatorHeight) == CHECKBOX_SIZE

    option = QStyleOptionButton()
    option.rect = QRect(0, 0, CHECKBOX_SIZE, CHECKBOX_SIZE)
    option.state = QStyle.StateFlag.State_Enabled | QStyle.StateFlag.State_On
    image = QImage(CHECKBOX_SIZE, CHECKBOX_SIZE, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    paint_rounded_checkbox(painter, option)
    style.drawPrimitive(
        QStyle.PrimitiveElement.PE_IndicatorCheckBox,
        option,
        painter,
        None,
    )
    painter.end()
    assert any(
        image.pixelColor(x, y).alpha() > 0
        for x in range(CHECKBOX_SIZE)
        for y in range(CHECKBOX_SIZE)
    )
