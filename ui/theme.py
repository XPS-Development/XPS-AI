"""Application-wide Fusion light theme: colors, stylesheet, and popup polish.

Call :func:`install_theme` once on the ``QApplication`` before creating widgets.
Rounded menus (and optionally combo popups) become translucent via
:class:`EditorStyle` so stylesheet ``border-radius`` shows through on Windows
and Linux. The same style draws rounded checkbox indicators for ``QCheckBox``
and item-view check columns.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import (
    QApplication,
    QMenu,
    QProxyStyle,
    QStyle,
    QStyleOption,
    QStyleOptionMenuItem,
    QWidget,
)

# Side panels sit slightly off-white; the plot stays pure white.
PANEL_BG = "#f7f7f7"
SURFACE = "#ffffff"
TEXT = "#000000"
TEXT_MUTED = "#666666"
TEXT_SUBTLE = "#888888"
TEXT_HEADER = "#555555"
TEXT_DISABLED = "#a0a0a0"
BORDER = "#d0d0d0"
BORDER_INPUT_SOFT = "#d8d8d8"
BORDER_HOVER = "#b8b8b8"
BORDER_SOFT = "#e0e0e0"
HAIRLINE = "#e8e8e8"
MENU_SEPARATOR = "#ebebeb"
SELECTION_BG = "#e4e4e4"
HOVER_BG = "#ececec"
BUTTON_HOVER_BG = "#f0f0f0"
BUTTON_DISABLED_BG = "#f3f3f3"
STATUS_BG = "#fafafa"
STATUS_BORDER = "#e5e5e5"
SPLITTER = "#c8c8c8"
SPLITTER_HOVER = "#a8a8a8"
POPUP_BORDER = "#a8a8a8"
SLIDER_TRACK = "#d0d0d0"
SLIDER_HANDLE = "#6e6e6e"
SLIDER_HANDLE_HOVER = "#555555"
SLIDER_HANDLE_PRESSED = "#444444"
SCROLLBAR_THUMB = "#c8c8c8"
SCROLLBAR_THUMB_HOVER = "#a8a8a8"

ITEM_RADIUS = 6
POPUP_RADIUS = 10
CHECKBOX_SIZE = 16
CHECKBOX_RADIUS = 4.0

_INSTALLED_STYLE: EditorStyle | None = None

APP_STYLESHEET = f"""
/* --- Trees (editorRows property selects hairlines vs spaced) --- */
QTreeView[editorRows="hairlines"],
QTreeView[editorRows="spaced"] {{
    background: {PANEL_BG};
    border: none;
    outline: none;
    show-decoration-selected: 0;
    selection-background-color: {SELECTION_BG};
    selection-color: {TEXT};
}}
QTreeView[editorRows="hairlines"]::item,
QTreeView[editorRows="spaced"]::item {{
    background: transparent;
    margin: 0px;
    border: none;
}}
QTreeView[editorRows="hairlines"]::item:hover,
QTreeView[editorRows="hairlines"]::item:hover:!selected,
QTreeView[editorRows="hairlines"]::item:selected,
QTreeView[editorRows="hairlines"]::item:selected:active,
QTreeView[editorRows="hairlines"]::item:selected:!active,
QTreeView[editorRows="spaced"]::item:hover,
QTreeView[editorRows="spaced"]::item:hover:!selected,
QTreeView[editorRows="spaced"]::item:selected,
QTreeView[editorRows="spaced"]::item:selected:active,
QTreeView[editorRows="spaced"]::item:selected:!active {{
    background: transparent;
    color: {TEXT};
}}
QTreeView[editorRows="hairlines"]::branch,
QTreeView[editorRows="spaced"]::branch {{
    background: transparent;
}}
QTreeView[editorRows="hairlines"]::item {{
    border-bottom: 1px solid {HAIRLINE};
    padding: 2px 4px;
}}
QTreeView[editorRows="spaced"]::item {{
    padding: 4px 4px;
}}
QTreeView[editorRows="hairlines"] QHeaderView::section {{
    background: {PANEL_BG};
    border: none;
    border-bottom: 1px solid {BORDER_SOFT};
    padding: 4px 6px;
    color: {TEXT_HEADER};
}}
QTreeView[editorRows="hairlines"]::indicator,
QTreeView[editorRows="spaced"]::indicator {{
    width: {CHECKBOX_SIZE}px;
    height: {CHECKBOX_SIZE}px;
}}

/* --- Checkboxes --- */
QCheckBox {{
    spacing: 8px;
    color: {TEXT};
}}
QCheckBox::indicator {{
    width: {CHECKBOX_SIZE}px;
    height: {CHECKBOX_SIZE}px;
}}

/* --- Combo boxes --- */
QComboBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: {ITEM_RADIUS}px;
    padding: 2px 8px;
    color: {TEXT};
    min-height: 20px;
}}
QComboBox:hover {{
    border: 1px solid {BORDER_HOVER};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 18px;
    border: none;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: {POPUP_RADIUS}px;
    padding: 4px;
    outline: none;
    selection-background-color: {SELECTION_BG};
    selection-color: {TEXT};
}}
QComboBoxPrivateContainer {{
    background: transparent;
    border: none;
}}

/* --- Menus --- */
QMenu {{
    background: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: {POPUP_RADIUS}px;
    padding: 4px;
}}
QMenu::item {{
    background: transparent;
    color: {TEXT};
    padding: 6px 36px 6px 12px;
    border-radius: {ITEM_RADIUS}px;
    margin: 1px 2px;
}}
QMenu::item:selected {{
    background: {SELECTION_BG};
    color: {TEXT};
}}
QMenu::item:disabled {{
    color: {TEXT_DISABLED};
}}
QMenu::item:checked {{
    font-weight: normal;
}}
QMenu::separator {{
    height: 1px;
    background: {MENU_SEPARATOR};
    margin: 4px 8px;
}}
QMenu::indicator {{
    width: 14px;
    height: 14px;
    margin-right: 4px;
}}
QMenu::right-arrow {{
    width: 10px;
    height: 10px;
    margin-right: 6px;
}}

/* --- Splitter --- */
QSplitter#MainSplitter::handle:horizontal {{
    background: {SPLITTER};
    width: 1px;
    margin: 0;
    padding: 0;
}}
QSplitter#MainSplitter::handle:horizontal:hover {{
    background: {SPLITTER_HOVER};
}}

/* --- Status bar --- */
QStatusBar {{
    background: {STATUS_BG};
    border-top: 1px solid {STATUS_BORDER};
    color: {TEXT_MUTED};
}}

/* --- Message boxes and dialog buttons --- */
QMessageBox {{
    background: {SURFACE};
}}
QMessageBox QLabel {{
    color: {TEXT};
    background: transparent;
}}
QMessageBox QPushButton,
QDialogButtonBox QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: {ITEM_RADIUS}px;
    padding: 4px 16px;
    min-height: 24px;
    color: {TEXT};
}}
QDialogButtonBox QPushButton {{
    min-width: 72px;
}}
QMessageBox QPushButton:hover,
QDialogButtonBox QPushButton:hover {{
    background: {BUTTON_HOVER_BG};
    border: 1px solid {BORDER_HOVER};
}}
QMessageBox QPushButton:pressed,
QDialogButtonBox QPushButton:pressed {{
    background: {SELECTION_BG};
}}
QMessageBox QPushButton:default,
QDialogButtonBox QPushButton:default {{
    background: {PANEL_BG};
    border: 1px solid {BORDER_HOVER};
}}
QMessageBox QPushButton:disabled,
QDialogButtonBox QPushButton:disabled {{
    color: {TEXT_DISABLED};
    background: {BUTTON_DISABLED_BG};
}}
QDialog {{
    background: {SURFACE};
    color: {TEXT};
}}

/* --- Side panels --- */
QWidget#SpectrumTreePanel,
QWidget#PropertiesPanel {{
    background: {PANEL_BG};
}}
QWidget#SpectrumTreePanel QLineEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER_INPUT_SOFT};
    border-radius: 4px;
    padding: 3px 6px;
}}
QWidget#SpectrumTreePanel QPushButton,
QWidget#PropertiesPanel QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 10px;
}}
QWidget#SpectrumTreePanel QPushButton:hover,
QWidget#PropertiesPanel QPushButton:hover {{
    background: {BUTTON_HOVER_BG};
}}
QWidget#PropertiesPanel QPushButton:disabled {{
    color: {TEXT_DISABLED};
    background: {BUTTON_DISABLED_BG};
}}

/* --- Expression editor popup (chrome only; tree uses editorRows) --- */
QFrame#ExprEditorPopup {{
    background: transparent;
    border: none;
}}
QLineEdit#ExprEditorField,
QLineEdit#ExprSearchField {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: {ITEM_RADIUS}px;
    padding: 4px 8px;
    color: {TEXT};
    min-height: 22px;
}}
QLineEdit#ExprEditorField:focus,
QLineEdit#ExprSearchField:focus {{
    border: 1px solid {BORDER_HOVER};
}}
QToolButton#ExprCopyButton {{
    background: transparent;
    border: none;
    padding: 2px;
    border-radius: 4px;
}}
QToolButton#ExprCopyButton:hover {{
    background: {BUTTON_HOVER_BG};
}}

/* --- Parameter value editor --- */
ParameterValueEditor {{
    background: transparent;
}}
QDoubleSpinBox#ParameterValueSpin {{
    background: transparent;
    border: none;
}}
QSlider#ParameterValueSlider::groove:horizontal {{
    height: 2px;
    background: {SLIDER_TRACK};
    border: none;
    border-radius: 1px;
    margin: 0 6px;
}}
QSlider#ParameterValueSlider::sub-page:horizontal,
QSlider#ParameterValueSlider::add-page:horizontal {{
    background: {SLIDER_TRACK};
    border: none;
    border-radius: 1px;
}}
QSlider#ParameterValueSlider::handle:horizontal {{
    width: 8px;
    height: 8px;
    margin: -3px 0;
    border: none;
    border-radius: 4px;
    background: {SLIDER_HANDLE};
}}
QSlider#ParameterValueSlider::handle:horizontal:hover {{
    background: {SLIDER_HANDLE_HOVER};
}}
QSlider#ParameterValueSlider::handle:horizontal:pressed {{
    background: {SLIDER_HANDLE_PRESSED};
}}

/* --- Inline editors / plot overlay --- */
QLineEdit#PropertiesFieldEdit {{
    background: transparent;
    border: none;
    padding: 1px 2px;
}}
QLabel#PlotCursorLabel, QLabel#PlotChiSquareLabel {{
    background-color: rgba(255, 255, 255, 0.8);
    padding: 2px 4px;
    border-radius: 2px;
}}

/* --- Scrollbars (Cursor-like thin floating pills) --- */
QScrollBar:vertical {{
    background: transparent;
    border: none;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {SCROLLBAR_THUMB};
    border: none;
    border-radius: 3px;
    min-height: 24px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{
    background: {SCROLLBAR_THUMB_HOVER};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
    width: 0;
    border: none;
    background: none;
}}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: none;
}}
QScrollBar:horizontal {{
    background: transparent;
    border: none;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {SCROLLBAR_THUMB};
    border: none;
    border-radius: 3px;
    min-width: 24px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {SCROLLBAR_THUMB_HOVER};
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    height: 0;
    width: 0;
    border: none;
    background: none;
}}
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: none;
}}
"""


def translucent_popups_supported() -> bool:
    """
    Return whether translucent rounded popups should be enabled.

    Set the environment variable ``XPS_AI_OPAQUE_POPUPS`` to any non-empty
    value to disable translucency (workaround for X11 without a compositor,
    where transparent corners otherwise paint black).

    Returns
    -------
    bool
        ``True`` when popups may use per-pixel alpha.
    """
    return not bool(os.environ.get("XPS_AI_OPAQUE_POPUPS"))


def make_translucent_popup(widget: QWidget) -> None:
    """
    Let a popup window show rounded corners through a transparent background.

    Call before the widget is first shown. Safe to call more than once.
    Do not call on an already-visible window: ``setWindowFlags`` hides it.

    Parameters
    ----------
    widget : QWidget
        Top-level popup (menu, custom popup frame).
    """
    # Windows needs FramelessWindowHint for per-pixel alpha, otherwise the area
    # outside the rounded chrome is black; the native drop shadow is rectangular
    # and reads as a second frame around rounded chrome.
    flags = widget.windowFlags()
    wanted = flags | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint
    if wanted != flags:
        widget.setWindowFlags(wanted)
    widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)


def _wants_rounded_popup(widget: QWidget) -> bool:
    """Return True when ``widget`` should get translucent rounded chrome."""
    if not translucent_popups_supported():
        return False
    return isinstance(widget, QMenu) or widget.inherits("QComboBoxPrivateContainer")


def paint_rounded_checkbox(painter: QPainter, option: QStyleOption) -> None:
    """
    Paint a rounded checkbox indicator into ``option.rect``.

    Used by :class:`EditorStyle` for ``QCheckBox`` and item-view check
    indicators so the same look appears everywhere.

    Parameters
    ----------
    painter : QPainter
        Active painter.
    option : QStyleOption
        Style option whose ``rect`` and ``state`` describe the indicator.
    """
    state = option.state
    enabled = bool(state & QStyle.StateFlag.State_Enabled)
    hovered = bool(state & QStyle.StateFlag.State_MouseOver) and enabled
    checked = bool(state & QStyle.StateFlag.State_On)
    partial = bool(state & QStyle.StateFlag.State_NoChange) and not checked

    rect = option.rect
    size = min(CHECKBOX_SIZE, rect.width(), rect.height())
    box_rect = QRect(
        rect.x() + max(0, (rect.width() - size) // 2),
        rect.y() + max(0, (rect.height() - size) // 2),
        size,
        size,
    )
    box = QRectF(box_rect)
    path = QPainterPath()
    path.addRoundedRect(box, CHECKBOX_RADIUS, CHECKBOX_RADIUS)

    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    if checked or partial:
        fill = QColor(TEXT if enabled else TEXT_DISABLED)
        painter.fillPath(path, fill)
        painter.setPen(Qt.PenStyle.NoPen)
    else:
        painter.fillPath(path, QColor(SURFACE))
        border = QColor(BORDER_HOVER if hovered else (BORDER if enabled else TEXT_DISABLED))
        pen = QPen(border)
        pen.setWidthF(1.25)
        painter.setPen(pen)
        painter.drawPath(path)

    if checked or partial:
        mark = QColor(SURFACE)
        pen = QPen(mark)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        x, y, w, h = box.x(), box.y(), box.width(), box.height()
        if checked:
            pen.setWidthF(1.8)
            painter.setPen(pen)
            painter.drawLine(
                QPointF(x + w * 0.28, y + h * 0.52),
                QPointF(x + w * 0.42, y + h * 0.68),
            )
            painter.drawLine(
                QPointF(x + w * 0.42, y + h * 0.68),
                QPointF(x + w * 0.72, y + h * 0.32),
            )
        else:
            pen.setWidthF(2.0)
            painter.setPen(pen)
            mid_y = box.center().y()
            inset = box.width() * 0.22
            painter.drawLine(
                QPointF(box.left() + inset, mid_y),
                QPointF(box.right() - inset, mid_y),
            )

    painter.restore()


class EditorStyle(QProxyStyle):
    """Fusion proxy: translucent menus and rounded checkbox indicators."""

    def __init__(self) -> None:
        """Initialize with the Fusion base style."""
        super().__init__("Fusion")

    def polish(  # ty: ignore[invalid-method-override]
        self, arg: QWidget | QPalette | QApplication
    ) -> QPalette | None:
        """
        Apply translucent flags to rounded popups, then forward to Fusion.

        Parameters
        ----------
        arg : QWidget or QPalette or QApplication
            Object to polish (Qt overload).

        Returns
        -------
        QPalette or None
            Palette when polishing a palette; otherwise ``None``.
        """
        if isinstance(arg, QWidget) and _wants_rounded_popup(arg):
            make_translucent_popup(arg)
        return super().polish(arg)

    def pixelMetric(
        self,
        metric: QStyle.PixelMetric,
        option: QStyleOption | None = None,
        widget: QWidget | None = None,
    ) -> int:
        """Return a uniform checkbox indicator size."""
        if metric in {
            QStyle.PixelMetric.PM_IndicatorWidth,
            QStyle.PixelMetric.PM_IndicatorHeight,
        }:
            return CHECKBOX_SIZE
        return super().pixelMetric(metric, option, widget)

    def drawPrimitive(
        self,
        element: QStyle.PrimitiveElement,
        option: QStyleOption,
        painter: QPainter,
        widget: QWidget | None = None,
    ) -> None:
        """Draw rounded checkboxes for widgets and item views."""
        if element in {
            QStyle.PrimitiveElement.PE_IndicatorCheckBox,
            QStyle.PrimitiveElement.PE_IndicatorItemViewItemCheck,
        }:
            paint_rounded_checkbox(painter, option)
            return
        super().drawPrimitive(element, option, painter, widget)

    def drawControl(
        self,
        element: QStyle.ControlElement,
        option: QStyleOption,
        painter: QPainter,
        widget: QWidget | None = None,
    ) -> None:
        """Draw menu shortcuts in a column on the right edge of the item."""
        if element == QStyle.ControlElement.CE_MenuItem and isinstance(
            option, QStyleOptionMenuItem
        ):
            label, sep, shortcut = option.text.partition("\t")
            if sep and shortcut:
                opt = QStyleOptionMenuItem(option)
                opt.text = label
                super().drawControl(element, opt, painter, widget)
                self._draw_menu_shortcut(opt, painter, shortcut)
                return
        super().drawControl(element, option, painter, widget)

    @staticmethod
    def _draw_menu_shortcut(
        option: QStyleOptionMenuItem,
        painter: QPainter,
        shortcut: str,
    ) -> None:
        """Paint ``shortcut`` vertically centered on the right of ``option``."""
        enabled = bool(option.state & QStyle.StateFlag.State_Enabled)
        role = QPalette.ColorRole.Text if enabled else QPalette.ColorRole.PlaceholderText
        painter.save()
        painter.setPen(option.palette.color(role))
        painter.drawText(
            option.rect.adjusted(12, 0, -12, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            shortcut,
        )
        painter.restore()


def build_palette(style: QStyle) -> QPalette:
    """
    Return the standard light Fusion palette for ``style``.

    Selection accents for trees and menus are set in the application
    stylesheet so text fields keep the platform selection color.

    Parameters
    ----------
    style : QStyle
        Style providing ``standardPalette`` (typically :class:`EditorStyle`).

    Returns
    -------
    QPalette
        Light application palette.
    """
    return style.standardPalette()


def install_theme(app: QApplication) -> EditorStyle:
    """
    Install Fusion light theme and the application stylesheet.

    Call once before creating any widgets. Keeps a Python reference to the
    proxy style so it is not garbage-collected.

    Parameters
    ----------
    app : QApplication
        Application instance.

    Returns
    -------
    EditorStyle
        Installed proxy style.
    """
    global _INSTALLED_STYLE
    style = EditorStyle()
    app.setStyle(style)
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    app.setPalette(build_palette(style))
    app.setStyleSheet(APP_STYLESHEET)
    _INSTALLED_STYLE = style
    return style
