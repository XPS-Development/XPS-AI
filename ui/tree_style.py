"""Shared editor-style look for spectrum and properties trees and menus."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QModelIndex, QPersistentModelIndex, QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPalette
from PySide6.QtWidgets import QMenu, QStyle, QStyleOptionViewItem, QTreeView, QWidget

if TYPE_CHECKING:
    from PySide6.QtWidgets import QComboBox

# Side panels sit slightly off-white; the plot stays pure white.
_PANEL_BG = "#f7f7f7"
_SELECTION_BG = "#e4e4e4"
_HOVER_BG = "#ececec"
_ITEM_RADIUS = 6
_MENU_RADIUS = "10px"
_ROW_INSET_X = 4
_ROW_INSET_Y = 1

_TREE_BASE = f"""
QTreeView {{
    background: {_PANEL_BG};
    border: none;
    outline: none;
    show-decoration-selected: 0;
}}
QTreeView::item {{
    background: transparent;
    margin: 0px;
    border: none;
}}
QTreeView::item:hover,
QTreeView::item:hover:!selected,
QTreeView::item:selected,
QTreeView::item:selected:active,
QTreeView::item:selected:!active {{
    background: transparent;
    color: #000000;
}}
QTreeView::branch {{
    background: transparent;
}}
"""

_ITEM_HAIRLINES = """
QTreeView::item {
    border-bottom: 1px solid #e8e8e8;
    padding: 2px 4px;
}
"""

_ITEM_SPACED = """
QTreeView::item {
    padding: 4px 4px;
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
    border-radius: {_MENU_RADIUS};
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
    border-radius: {_MENU_RADIUS};
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
QMenu::item:disabled {{
    color: #a0a0a0;
}}
QMenu::item:checked {{
    font-weight: normal;
}}
QMenu::separator {{
    height: 1px;
    background: #ebebeb;
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
"""


def _as_model_index(index: QModelIndex | QPersistentModelIndex) -> QModelIndex:
    """Return a plain ``QModelIndex`` from a possibly persistent index."""
    model = index.model()
    if model is None:
        return QModelIndex()
    return model.index(index.row(), index.column(), index.parent())


class EditorTreeView(QTreeView):
    """
    Tree view with one continuous rounded selection/hover strip per row.

    Stylesheet selection paints per cell and looks torn across columns and the
    branch gutter; this view draws a single rounded rect behind the row instead.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hover_row_index = QPersistentModelIndex()
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setSelectionBehavior(QTreeView.SelectionBehavior.SelectRows)

    def drawRow(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Paint a full-width rounded row chrome, then the cell contents."""
        idx = _as_model_index(index)
        if idx.isValid():
            self._paint_row_background(painter, option, idx)
        super().drawRow(painter, option, index)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Track the hovered row for rounded hover chrome."""
        self._set_hover_row(self.indexAt(event.position().toPoint()))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        """Clear hover chrome when the pointer leaves the view."""
        self._set_hover_row(QModelIndex())
        super().leaveEvent(event)

    def _paint_row_background(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        """Draw selection or hover chrome as one rounded rectangle.

        Uses ``option.rect`` for vertical placement so we never call
        ``visualRect`` during painting (which can segfault in Qt).
        """
        sel = self.selectionModel()
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        if not selected and sel is not None and index.isValid():
            selected = sel.isRowSelected(index.row(), index.parent())
        hovered = self._hover_row_index.isValid() and self._same_row(
            _as_model_index(self._hover_row_index), index
        )
        if not selected and not hovered:
            return
        height = max(0, option.rect.height() - 2 * _ROW_INSET_Y)
        width = max(0, self.viewport().width() - 2 * _ROW_INSET_X)
        if height <= 0 or width <= 0:
            return
        rect = QRectF(
            float(_ROW_INSET_X),
            float(option.rect.y() + _ROW_INSET_Y),
            float(width),
            float(height),
        )
        color = QColor(_SELECTION_BG if selected else _HOVER_BG)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(rect, float(_ITEM_RADIUS), float(_ITEM_RADIUS))
        painter.restore()

    def _set_hover_row(self, index: QModelIndex) -> None:
        """Update which row shows hover chrome and schedule a repaint."""
        new = index.sibling(index.row(), 0) if index.isValid() else QModelIndex()
        old = (
            _as_model_index(self._hover_row_index)
            if self._hover_row_index.isValid()
            else QModelIndex()
        )
        if self._same_row(old, new):
            return
        self._hover_row_index = (
            QPersistentModelIndex(new) if new.isValid() else QPersistentModelIndex()
        )
        # Full viewport update avoids visualRect on possibly-stale indexes.
        self.viewport().update()

    @staticmethod
    def _same_row(a: QModelIndex, b: QModelIndex) -> bool:
        """Return True when both indexes point at the same tree row."""
        if not a.isValid() and not b.isValid():
            return True
        if not a.isValid() or not b.isValid():
            return False
        return a.row() == b.row() and a.parent() == b.parent()


def apply_editor_tree_style(
    view: QTreeView,
    *,
    with_header: bool = False,
    row_separators: bool = True,
) -> None:
    """
    Apply editor-style panel chrome for trees with rounded row selection.

    Prefer :class:`EditorTreeView` so selection/hover paint as one continuous
    rounded strip. Plain ``QTreeView`` still gets transparent cell fills.

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


def _style_one_menu(menu: QMenu) -> None:
    """Apply palette and stylesheet to a single menu popup."""
    menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    palette = menu.palette()
    palette.setColor(QPalette.ColorRole.Highlight, QColor(_SELECTION_BG))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    menu.setPalette(palette)
    menu.setStyleSheet(_MENU_STYLE)


def apply_editor_menu_style(menu: QMenu) -> None:
    """
    Style a popup menu like the model picker (rounded chrome, gray selection).

    Also styles nested submenus already attached via ``addMenu``.

    Parameters
    ----------
    menu : QMenu
        Menu to style.
    """
    _style_one_menu(menu)
    for child in menu.findChildren(QMenu):
        _style_one_menu(child)
