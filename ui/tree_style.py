"""Tree view with continuous rounded selection/hover row chrome.

Application colors and stylesheets live in :mod:`ui.theme`. Call
:func:`apply_editor_tree_style` to select hairline vs spaced row padding via
the ``editorRows`` dynamic property.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QModelIndex, QPersistentModelIndex, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter
from PySide6.QtWidgets import QStyle, QStyleOptionViewItem, QTreeView, QWidget

from . import theme

_ROW_INSET_X = 4
_ROW_INSET_Y = 1


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
        self._hover_rect = QRect()
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

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        """Keep the cached hover rect aligned after the viewport scrolls."""
        super().scrollContentsBy(dx, dy)
        self._refresh_hover_rect()

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
        color = QColor(theme.SELECTION_BG if selected else theme.HOVER_BG)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(rect, float(theme.ITEM_RADIUS), float(theme.ITEM_RADIUS))
        painter.restore()

    def _set_hover_row(self, index: QModelIndex) -> None:
        """Update which row shows hover chrome and schedule a targeted repaint.

        Invalidates only the previous and new row strips. The previous strip
        uses a cached ``QRect`` (never ``visualRect`` on a stale index). The
        new strip uses ``visualRect`` on a fresh ``indexAt`` / sibling index,
        which is safe outside of paint handlers.
        """
        new = index.sibling(index.row(), 0) if index.isValid() else QModelIndex()
        old = (
            _as_model_index(self._hover_row_index)
            if self._hover_row_index.isValid()
            else QModelIndex()
        )
        if self._same_row(old, new):
            return
        viewport = self.viewport()
        old_rect = QRect(self._hover_rect)
        if not old_rect.isNull():
            viewport.update(old_rect)
        self._hover_row_index = (
            QPersistentModelIndex(new) if new.isValid() else QPersistentModelIndex()
        )
        self._hover_rect = self._row_chrome_rect(new) if new.isValid() else QRect()
        if not self._hover_rect.isNull():
            viewport.update(self._hover_rect)

    def _refresh_hover_rect(self) -> None:
        """Recompute ``_hover_rect`` after layout shifts (e.g. scroll)."""
        if not self._hover_row_index.isValid():
            self._hover_rect = QRect()
            return
        idx = _as_model_index(self._hover_row_index)
        self._hover_rect = self._row_chrome_rect(idx) if idx.isValid() else QRect()

    def _row_chrome_rect(self, index: QModelIndex) -> QRect:
        """Return the full-width viewport rect covering the row chrome."""
        if not index.isValid():
            return QRect()
        cell = self.visualRect(index)
        if cell.isNull():
            return QRect()
        # Pad 1px for antialiased rounded corners beyond the cell bounds.
        return QRect(0, cell.y() - 1, self.viewport().width(), cell.height() + 2)

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
    row_separators: bool = True,
) -> None:
    """
    Mark a tree for editor-style row chrome via the ``editorRows`` property.

    Prefer :class:`EditorTreeView` so selection/hover paint as one continuous
    rounded strip. The application stylesheet in :mod:`ui.theme` styles trees
    that have this property set.

    Parameters
    ----------
    view : QTreeView
        Tree to style.
    row_separators : bool, optional
        When True, draw light hairlines between rows; otherwise use slightly
        taller row padding (spectrum tree).
    """
    view.setProperty("editorRows", "hairlines" if row_separators else "spaced")
    style = view.style()
    if style is not None:
        for widget in (view, view.header()):
            style.unpolish(widget)
            style.polish(widget)
