"""Delegate that paints a display name with an optional gray ID suffix."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, Qt
from PySide6.QtGui import QColor, QIcon, QPainter
from PySide6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionViewItem

from .component_colors import ID_SUFFIX_HEX

# Custom roles used by trees that show truncated IDs next to labels.
ObjectIdRole = Qt.ItemDataRole.UserRole
ObjectIdPrefixRole = Qt.ItemDataRole.UserRole + 1
ComponentColorRole = Qt.ItemDataRole.UserRole + 2

_SWATCH_SIZE = 10
_SWATCH_GAP = 6


class NameWithIdDelegate(QStyledItemDelegate):
    """
    Paint optional color swatch, ``DisplayRole`` text, then a gray truncated id.

    Color comes from ``ComponentColorRole`` (hex string). Id suffix comes from
    ``ObjectIdPrefixRole``.
    """

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Draw the standard item chrome, then swatch + name + gray id suffix."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        id_prefix = index.data(ObjectIdPrefixRole)
        has_id = isinstance(id_prefix, str) and bool(id_prefix)
        color_value = index.data(ComponentColorRole)
        swatch_color = QColor(color_value) if isinstance(color_value, str) else QColor()

        text = opt.text
        opt.text = ""
        # Keep decoration empty; we draw the swatch ourselves for consistent layout.
        opt.icon = QIcon()
        opt.features &= ~QStyleOptionViewItem.ViewItemFeature.HasDecoration

        widget = opt.widget
        style = widget.style() if widget is not None else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
        else:
            super().paint(painter, option, index)
            return

        painter.save()
        painter.setClipRect(opt.rect)

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText,
            opt,
            widget,
        )
        if not text_rect.isValid():
            text_rect = opt.rect

        x = text_rect.x()
        if swatch_color.isValid():
            y = text_rect.y() + max(0, (text_rect.height() - _SWATCH_SIZE) // 2)
            painter.fillRect(x, y, _SWATCH_SIZE, _SWATCH_SIZE, swatch_color)
            x += _SWATCH_SIZE + _SWATCH_GAP

        font = opt.font
        painter.setFont(font)
        metrics = painter.fontMetrics()

        if opt.state & QStyle.StateFlag.State_Selected:
            painter.setPen(opt.palette.highlightedText().color())
        else:
            painter.setPen(opt.palette.text().color())

        name = text or ""
        name_width = metrics.horizontalAdvance(name) if name else 0
        name_rect = text_rect.adjusted(x - text_rect.x(), 0, 0, 0)
        painter.drawText(
            name_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            name,
        )

        if has_id:
            gap = metrics.horizontalAdvance(" ")
            suffix = f" {id_prefix}"
            suffix_x = x + name_width + gap
            if opt.state & QStyle.StateFlag.State_Selected:
                sel = QColor(opt.palette.highlightedText().color())
                sel.setAlpha(180)
                painter.setPen(sel)
            else:
                painter.setPen(QColor(ID_SUFFIX_HEX))
            suffix_rect = text_rect.adjusted(suffix_x - text_rect.x(), 0, 0, 0)
            painter.drawText(
                suffix_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                suffix,
            )

        painter.restore()

    def sizeHint(
        self,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> Any:
        """Widen the hint for swatch and truncated IDs."""
        hint = super().sizeHint(option, index)
        extra = 0
        if isinstance(index.data(ComponentColorRole), str):
            extra += _SWATCH_SIZE + _SWATCH_GAP
        id_prefix = index.data(ObjectIdPrefixRole)
        if isinstance(id_prefix, str) and id_prefix:
            metrics = option.fontMetrics
            extra += metrics.horizontalAdvance(f" {id_prefix}")
        if extra:
            hint.setWidth(hint.width() + extra)
        return hint
