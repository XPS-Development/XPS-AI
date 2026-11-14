"""Sortable, copyable presentation of immutable post-processing table projections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import (
    QAbstractTableModel,
    QIdentityProxyModel,
    QModelIndex,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
)
from PySide6.QtGui import QColor, QKeyEvent, QKeySequence
from PySide6.QtWidgets import QAbstractItemView, QApplication, QTableView

from ui.component_colors import color_for_component
from ui.theme import HOVER_BG, TEXT_MUTED

if TYPE_CHECKING:
    from core.dto import AnalysisRow, AnalysisTable

AMBIGUOUS_BG = "#fff1cf"


class AnalysisTableModel(QAbstractTableModel):
    """Render source values without performing scientific calculations."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.table: AnalysisTable | None = None

    def set_table(self, table: AnalysisTable) -> None:
        """Replace the projection; sorting remains a property of the proxy."""
        self.beginResetModel()
        self.table = table
        self.endResetModel()

    def clear(self) -> None:
        """Clear a workspace when its document source is no longer available."""
        self.beginResetModel()
        self.table = None
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:
        """Return the number of projected rows."""
        return (
            0
            if (parent is not None and parent.isValid()) or self.table is None
            else len(self.table.rows)
        )

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:
        """Return the number of projected columns."""
        return (
            0
            if (parent is not None and parent.isValid()) or self.table is None
            else len(self.table.columns)
        )

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        """Show units beside labels and retain column identities in tooltips."""
        if self.table is None or orientation != Qt.Orientation.Horizontal:
            return super().headerData(section, orientation, role)
        column = self.table.columns[section]
        if role == Qt.ItemDataRole.DisplayRole:
            label = column.label
            if column.track_id and " — " in label:
                label = label.replace(" / ", "\n", 1).replace(" — ", "\n", 1)
            return f"{label} ({column.unit})" if column.unit and column.unit != "area" else label
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"{column.label}\n{column.id_}"
        return None

    def data(
        self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        """Return display, raw sort and status information for one cell."""
        if not index.isValid() or self.table is None:
            return None
        cell = self.table.rows[index.row()].cells[index.column()]
        column = self.table.columns[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return (
                ""
                if cell.value is None
                else f"{cell.value:.6g}"
                if isinstance(cell.value, float)
                else str(cell.value)
            )
        if role == Qt.ItemDataRole.UserRole:
            return cell.value
        if role == Qt.ItemDataRole.ToolTipRole:
            status = ": ".join(value for value in (cell.status, cell.reason) if value)
            return "\n".join(
                (status, *(f"{s.spectrum_id} / {s.component_id}" for s in cell.sources))
            )
        if role == Qt.ItemDataRole.BackgroundRole:
            if cell.status == "ambiguous":
                return QColor(AMBIGUOUS_BG)
            if cell.status == "missing":
                return QColor(HOVER_BG)
        if role == Qt.ItemDataRole.ForegroundRole:
            if cell.value is None:
                return QColor(TEXT_MUTED)
            if column.anchor_id:
                return QColor(color_for_component(component_id=column.anchor_id))
        return None


class AnalysisTableView(QTableView):
    """Keep row navigation stable through proxy sorting and copy selected cells as TSV."""

    def __init__(self, parent: Any = None, *, trend: bool = False, sortable: bool = True) -> None:
        super().__init__(parent)
        self._trend = trend
        self._details = False
        self.setShowGrid(False)
        self.setObjectName("PostProcessingTable")
        self.source = AnalysisTableModel(self)
        self.proxy = QSortFilterProxyModel(self) if sortable else QIdentityProxyModel(self)
        self.proxy.setSourceModel(self.source)
        if isinstance(self.proxy, QSortFilterProxyModel):
            self.proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.setModel(self.proxy)
        self.setSortingEnabled(sortable)
        self.horizontalHeader().setSectionsClickable(sortable)
        self.horizontalHeader().setSortIndicatorShown(sortable)
        if sortable:
            self.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().hide()
        self.source.modelReset.connect(self._identity_columns)

    def _identity_columns(self) -> None:
        table = self.source.table
        if table is not None:
            self.horizontalHeader().setMinimumHeight(
                56 if not self._trend and any(c.track_id for c in table.columns) else 28
            )
            for index, column in enumerate(table.columns):
                hidden = column.id_ in {
                    "row_id",
                    "region_id",
                    "component_id",
                    "track_id",
                    "region",
                    "component",
                    "track",
                    "track_name",
                }
                if self._trend and not self._details:
                    hidden |= column.id_ not in {
                        "step",
                        "spectrum",
                        "position",
                        "area",
                        "sig",
                        "fraction",
                        "link",
                    }
                self.setColumnHidden(index, hidden)

    def set_details(self, enabled: bool) -> None:
        """Show additional scientific fields while keeping internal identities hidden."""
        self._details = enabled
        self._identity_columns()
        self.resizeColumnsToContents()

    def source_row(self, index: QModelIndex) -> AnalysisRow | None:
        """Resolve the original spectrum/component from a visible sorted row."""
        if not index.isValid() or self.source.table is None:
            return None
        return self.source.table.rows[self.proxy.mapToSource(index).row()]

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Copy a rectangular selection, retaining holes as empty TSV cells."""
        if event.matches(QKeySequence.StandardKey.Copy):
            indices = self.selectedIndexes()
            if indices:
                selected = {
                    (index.row(), index.column()): "" if index.data() is None else index.data()
                    for index in indices
                }
                rows = range(min(i.row() for i in indices), max(i.row() for i in indices) + 1)
                columns = [
                    col
                    for col in range(
                        min(i.column() for i in indices), max(i.column() for i in indices) + 1
                    )
                    if not self.isColumnHidden(col)
                ]
                QApplication.clipboard().setText(
                    "\n".join(
                        "\t".join(str(selected.get((row, col), "")) for col in columns)
                        for row in rows
                    )
                )
            event.accept()
            return
        super().keyPressEvent(event)
