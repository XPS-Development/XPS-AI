"""Shared editor-style look for spectrum and properties trees."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QColor, QPalette

if TYPE_CHECKING:
    from PySide6.QtWidgets import QTreeView

_TREE_BASE = """
QTreeView {
    background: #ffffff;
    border: none;
    outline: none;
}
QTreeView::item:hover:!selected {
    background: #f5f5f5;
}
QTreeView::item:selected,
QTreeView::item:selected:active {
    background: #e8e8e8;
    color: #000000;
}
"""

_ITEM_HAIRLINES = """
QTreeView::item {
    border-bottom: 1px solid #eeeeee;
    padding: 2px 0;
}
"""

_ITEM_SPACED = """
QTreeView::item {
    padding: 4px 0;
}
"""

_HEADER_STYLE = """
QHeaderView::section {
    background: #fafafa;
    border: none;
    border-bottom: 1px solid #e0e0e0;
    padding: 4px 6px;
    color: #555555;
}
"""


def apply_editor_tree_style(
    view: QTreeView,
    *,
    with_header: bool = False,
    row_separators: bool = True,
) -> None:
    """
    Apply gray selection with black text (modern editor look).

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
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#e8e8e8"))
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
