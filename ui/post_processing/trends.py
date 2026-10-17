"""Interactive pyqtgraph trends from shared analytical cells, without interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QListWidget, QListWidgetItem, QWidget

from ui.component_colors import color_for_component
from ui.plot.plot_area import DoubleClickAutoRangeViewBox

if TYPE_CHECKING:
    from core.dto import AnalysisTable


class TrendWidget(QWidget):
    """Separate measurement units into panels and leave missing values as gaps."""

    sourceActivated = Signal(str, object)
    selectionChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.columns = QListWidget()
        self.columns.setObjectName("PostProcessingTrendColumns")
        self.columns.setMaximumWidth(240)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.columns)
        layout.addWidget(self.graphics, 1)
        self._table: AnalysisTable | None = None
        self._series_id: str | None = None
        self.columns.itemChanged.connect(self._selection_changed)

    def set_table(
        self,
        series_id: str,
        table: AnalysisTable,
        *,
        selected_columns: tuple[str, ...] | None = None,
    ) -> None:
        """Refresh plotted values while retaining checked columns for the current series."""
        checked = {
            self.columns.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.columns.count())
            if self.columns.item(i).checkState() == Qt.CheckState.Checked
        }
        previous = {
            self.columns.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.columns.count())
        }
        if self._series_id != series_id:
            checked, previous = set(), set()
        if selected_columns is not None:
            checked = set(selected_columns)
            previous = {column.id_ for column in table.columns}
        self._series_id, self._table = series_id, table
        self.columns.blockSignals(True)
        self.columns.clear()
        for column in table.columns[2:]:
            if not column.unit:
                continue
            item = QListWidgetItem(f"{column.label} ({column.unit})")
            item.setData(Qt.ItemDataRole.UserRole, column.id_)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if column.id_ in checked or column.id_ not in previous
                else Qt.CheckState.Unchecked
            )
            self.columns.addItem(item)
        self.columns.blockSignals(False)
        self._draw()

    def _selection_changed(self) -> None:
        self._draw()
        self.selectionChanged.emit(
            tuple(
                self.columns.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.columns.count())
                if self.columns.item(i).checkState() == Qt.CheckState.Checked
            )
        )

    def _draw(self) -> None:
        self.graphics.ci.clear()
        table = self._table
        if table is None:
            return
        checked = {
            self.columns.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.columns.count())
            if self.columns.item(i).checkState() == Qt.CheckState.Checked
        }
        plots: dict[str, Any] = {}
        for column_index, column in enumerate(table.columns[2:], 2):
            if column.id_ not in checked:
                continue
            if column.unit not in plots:
                plot = self.graphics.ci.addPlot(
                    row=len(plots), col=0, viewBox=DoubleClickAutoRangeViewBox(enableMenu=False)
                )
                plot.setLabel("bottom", "Row" if any(row.id_ for row in table.rows) else "Step")
                plot.setLabel("left", column.unit)
                plot.showGrid(x=True, y=True, alpha=0.3)
                plot.addLegend()
                plots[column.unit] = plot
            plot = plots[column.unit]
            color = color_for_component(component_id=column.anchor_id or column.id_)
            cells = [row.cells[column_index] for row in table.rows]
            x = [row.step for row in table.rows]
            y = np.array([np.nan if cell.value is None else float(cell.value) for cell in cells])
            plot.plot(x, y, connect="finite", pen=pg.mkPen(color), name=column.label)
            spots = [
                {
                    "pos": (row.step, float(cell.value)),
                    "data": (row.id_ or row.spectrum_id, column.id_),
                    "symbol": "t" if cell.status == "ambiguous" else "o",
                    "brush": "#e6a817" if cell.status == "ambiguous" else color,
                }
                for row, cell in zip(table.rows, cells, strict=True)
                if cell.value is not None
            ]
            scatter = pg.ScatterPlotItem(spots, size=8, pen=pg.mkPen(color))
            scatter.sigClicked.connect(self._clicked)
            plot.addItem(scatter)

    def _clicked(self, _item: Any, points: Any) -> None:
        if points:
            spectrum_id, track_id = points[0].data()
            self.sourceActivated.emit(spectrum_id, track_id)
