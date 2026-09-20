"""Shared editor-style look for spectrum and properties trees."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette

if TYPE_CHECKING:
    from PySide6.QtWidgets import QComboBox, QMenu, QTreeView

# Side panels sit slightly off-white; the plot stays pure white.
_PANEL_BG = "#f7f7f7"
_SELECTION_BG = "#e4e4e4"
_HOVER_BG = "#ececec"

_TREE_BASE = f"""
QTreeView {{
    background: {_PANEL_BG};
    border: none;
    outline: none;
    show-decoration-selected: 1;
}}
QTreeView::item:hover:!selected {{
    background: {_HOVER_BG};
}}
QTreeView::item:selected,
QTreeView::item:selected:active {{
    background: {_SELECTION_BG};
    color: #000000;
}}
"""

_ITEM_HAIRLINES = """
QTreeView::item {
    border-bottom: 1px solid #e8e8e8;
    padding: 2px 0;
}
"""

_ITEM_SPACED = """
QTreeView::item {
    padding: 4px 0;
}
"""

_HEADER_STYLE = f"""
QHeaderView::section {{
    background: {_PANEL_BG};
    border: none;
    border-bottom: 1px solid #e0e0e0;
    padding: 4px 6px;
    color: #555555;
}}
"""

_COMBO_STYLE = f"""
QComboBox {{
    background: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    padding: 2px 8px;
    color: #000000;
    min-height: 20px;
}}
QComboBox:hover {{
    border: 1px solid #b8b8b8;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 18px;
    border: none;
}}
QComboBox QAbstractItemView {{
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 4px;
    outline: none;
    selection-background-color: {_SELECTION_BG};
    selection-color: #000000;
}}
"""

_MENU_STYLE = f"""
QMenu {{
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 4px;
}}
QMenu::item {{
    background: transparent;
    color: #000000;
    padding: 6px 28px 6px 12px;
    border-radius: 6px;
    margin: 1px 2px;
}}
QMenu::item:selected {{
    background: {_SELECTION_BG};
    color: #000000;
}}
QMenu::item:checked {{
    font-weight: normal;
}}
QMenu::indicator {{
    width: 14px;
    height: 14px;
    margin-right: 4px;
}}
"""


def apply_editor_tree_style(
    view: QTreeView,
    *,
    with_header: bool = False,
    row_separators: bool = True,
) -> None:
    """
    Apply editor-style panel chrome with continuous gray row selection.

    Parameters
    ----------
    view : QTreeView
        Tree to style.
    with_header : bool, optional
        When True, also style the header sections (properties panel).
    row_separators : bool, optional
        When True, draw light hairlines between rows; otherwise use slightly
        taller row padding (spectrum tree).
    """
    palette = view.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor(_PANEL_BG))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(_SELECTION_BG))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
    view.setPalette(palette)
    parts = [_TREE_BASE]
    if row_separators:
        parts.append(_ITEM_HAIRLINES)
    else:
        parts.append(_ITEM_SPACED)
    if with_header:
        parts.append(_HEADER_STYLE)
    view.setStyleSheet("".join(parts))


def apply_editor_combo_style(combo: QComboBox) -> None:
    """
    Style a combo box and its popup with rounded chrome and gray selection.

    Parameters
    ----------
    combo : QComboBox
        Combo box to style.
    """
    palette = combo.palette()
    palette.setColor(QPalette.ColorRole.Highlight, QColor(_SELECTION_BG))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
    combo.setPalette(palette)
    combo.setStyleSheet(_COMBO_STYLE)
    view = combo.view()
    if view is not None:
        view_palette = view.palette()
        view_palette.setColor(QPalette.ColorRole.Highlight, QColor(_SELECTION_BG))
        view_palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
        view_palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        view.setPalette(view_palette)


def apply_editor_menu_style(menu: QMenu) -> None:
    """
    Style a popup menu like a Cursor dropdown (rounded, gray selection).

    Parameters
    ----------
    menu : QMenu
        Menu to style.
    """
    menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    palette = menu.palette()
    palette.setColor(QPalette.ColorRole.Highlight, QColor(_SELECTION_BG))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    menu.setPalette(palette)
    menu.setStyleSheet(_MENU_STYLE)
