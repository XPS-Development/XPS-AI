"""Series/trend review and independent cross-series analytical workspaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.component_colors import color_for_component
from ui.post_processing.links import LinkEditor
from ui.post_processing.metric_dialog import MetricDialog
from ui.post_processing.series_dialog import SeriesDialog
from ui.post_processing.source_dialog import SourceDialog
from ui.post_processing.table import AnalysisTableView
from ui.post_processing.trends import TrendWidget

if TYPE_CHECKING:
    from core.dto import SourceLocation
    from ui.controller import ControllerWrapper


class SeriesPanel(QWidget):
    """Manage saved series through controller commands while retaining selection."""

    seriesSelected = Signal(object)
    tableSelected = Signal(object)

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PostProcessingSeriesPanel")
        self._controller = controller
        self.list = QListWidget()
        self.list.setObjectName("PostProcessingSeriesList")
        self._generation = controller.query.document_generation
        self._selected: str | None = None
        self._table_id: str | None = None
        self._kind = "series"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        row = QHBoxLayout()
        for name, callback in (("New", self._new), ("Edit", self._edit), ("Delete", self._delete)):
            button = QPushButton(name)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addWidget(QLabel("Spectrum series"))
        layout.addLayout(row)
        layout.addWidget(self.list, 1)
        layout.addWidget(QLabel("Analytical tables"))
        table_buttons = QHBoxLayout()
        for label, callback in (
            ("New", self._new_table),
            ("Sources…", self._edit_table),
            ("Delete", self._delete_table),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            table_buttons.addWidget(button)
        layout.addLayout(table_buttons)
        self.tables = QListWidget()
        self.tables.setObjectName("AnalyticalTableList")
        self.tables.currentItemChanged.connect(self._table_changed)
        self.tables.itemClicked.connect(self._table_changed)
        self.list.itemClicked.connect(self._selected_changed)
        for widget, table in ((self.list, False), (self.tables, True)):
            widget.setEditTriggers(
                QAbstractItemView.EditTrigger.DoubleClicked
                | QAbstractItemView.EditTrigger.EditKeyPressed
            )
            widget.itemChanged.connect(
                lambda item, table=table: self._rename_item(item, table=table)
            )
        layout.addWidget(self.tables, 1)
        self.list.currentItemChanged.connect(self._selected_changed)
        controller.postProcessingChanged.connect(self.refresh, Qt.ConnectionType.QueuedConnection)
        self.refresh()

    def refresh(self) -> None:
        """Refresh saved labels and keep the current series when it still exists."""
        if self._generation != self._controller.query.document_generation:
            self._selected = None
            self._table_id = None
            self._kind = "series"
            self._generation = self._controller.query.document_generation
        ids = self._controller.query.get_post_processing_series_ids()
        tids = self._controller.query.get_analytical_table_ids()
        if self._kind == "table" and self._table_id not in tids:
            self._kind = "series"
        selected = self._selected if self._selected in ids else next(iter(ids), None)
        self.list.blockSignals(True)
        self.list.clear()
        for sid in ids:
            dto = self._controller.query.get_post_processing_series(sid)
            item = QListWidgetItem(dto.series.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setData(Qt.ItemDataRole.UserRole, sid)
            item.setToolTip("Matching needs updating" if dto.stale else dto.series.name)
            self.list.addItem(item)
            if self._kind == "series" and sid == selected:
                self.list.setCurrentItem(item)
        self.list.blockSignals(False)
        self._selected = selected
        self.tables.blockSignals(True)
        self.tables.clear()
        for tid in tids:
            dto = self._controller.query.get_analytical_table(tid)
            item = QListWidgetItem(dto.state.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(
                "Sources need updating" if dto.stale or dto.conflicts else dto.state.name
            )
            item.setData(Qt.ItemDataRole.UserRole, tid)
            self.tables.addItem(item)
            if self._kind == "table" and tid == self._table_id:
                self.tables.setCurrentItem(item)
        self.tables.blockSignals(False)
        if self._kind == "table" and self._table_id in tids:
            self.tableSelected.emit(self._table_id)
        else:
            self._kind = "series"
            self.seriesSelected.emit(selected)

    def _selected_changed(self) -> None:
        item = self.list.currentItem()
        self._selected = None if item is None else item.data(Qt.ItemDataRole.UserRole)
        self._kind = "series"
        with QSignalBlocker(self.tables):
            self.tables.setCurrentRow(-1)
        self.seriesSelected.emit(self._selected)

    def _rename_item(self, item: QListWidgetItem, *, table: bool) -> None:
        identity = item.data(Qt.ItemDataRole.UserRole)
        old = (
            self._controller.query.get_analytical_table(identity).state.name
            if table
            else self._controller.query.get_post_processing_series(identity).series.name
        )
        name = item.text().strip()
        widget = self.tables if table else self.list
        with QSignalBlocker(widget):
            item.setText(name or old)
        if not name or name == old:
            return
        if table:
            self._controller.edit_analytical_table(identity, name=name)
        else:
            self._controller.edit_post_processing_series(identity, name=name)

    def _new(self) -> None:
        dialog = SeriesDialog(self._controller, self)
        if dialog.exec():
            self._selected = self._controller.create_post_processing_series(
                dialog.name_edit.text(), dialog.spectrum_ids()
            )
            self._kind = "series"
            self.refresh()

    def _edit(self) -> None:
        if self._selected is None:
            return
        series = self._controller.query.get_post_processing_series(self._selected).series
        dialog = SeriesDialog(
            self._controller, self, name=series.name, spectrum_ids=series.spectrum_ids
        )
        if dialog.exec():
            self._controller.edit_post_processing_series(
                self._selected, name=dialog.name_edit.text(), spectrum_ids=dialog.spectrum_ids()
            )

    def _delete(self) -> None:
        if self._selected is not None:
            self._controller.remove_post_processing_series(self._selected)

    def select_table(self, table_id: str) -> None:
        """Open a created table while retaining the selected series for later return."""
        self._kind, self._table_id = "table", table_id
        self.refresh()

    def _table_changed(self) -> None:
        item = self.tables.currentItem()
        if item is not None:
            self._kind = "table"
            with QSignalBlocker(self.list):
                self.list.setCurrentRow(-1)
            self._table_id = item.data(Qt.ItemDataRole.UserRole)
            self.tableSelected.emit(self._table_id)

    def _new_table(self) -> None:
        dialog = SourceDialog(self._controller, self)
        if dialog.exec():
            self.select_table(
                self._controller.create_analytical_table(
                    dialog.name_edit.text(), dialog.selections()
                )
            )

    def _edit_table(self) -> None:
        if self._table_id not in self._controller.query.get_analytical_table_ids():
            return
        state = self._controller.query.get_analytical_table(self._table_id).state
        dialog = SourceDialog(
            self._controller,
            self,
            name=state.name,
            selected=tuple((s.series_id, s.track_id) for s in state.sources),
        )
        if dialog.exec():
            self._controller.edit_analytical_table(
                state.id_, name=dialog.name_edit.text(), selections=dialog.selections()
            )

    def _delete_table(self) -> None:
        if self._table_id in self._controller.query.get_analytical_table_ids():
            self._controller.remove_analytical_table(self._table_id)


class PostProcessingPanel(QWidget):
    """Keep series verification separate from independent analytical tables."""

    sourceActivated = Signal(str, object, object)
    tableCreated = Signal(str)

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setObjectName("PostProcessingPanel")
        self.series_id = None
        self.table_id = None
        self._selected_tracks = {}
        self._kind = "series"
        self._series_status = ""
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.stack = QStackedWidget()
        self.series_page = QWidget()
        self.analysis_page = QWidget()
        self.stack.addWidget(self.series_page)
        self.stack.addWidget(self.analysis_page)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.stack, 1)
        self.empty_page = QWidget()
        empty_layout = QVBoxLayout(self.empty_page)
        empty_layout.addStretch()
        heading = QLabel("Peak trends and analytical tables")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(heading)
        hint = QLabel(
            "Create a series from decomposed spectra, then match its peaks.\nSelect a series or table in the sidebar to continue."
        )
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(hint)
        create = QPushButton("Create series…")
        create.clicked.connect(self._create_series)
        empty_layout.addWidget(create, 0, Qt.AlignmentFlag.AlignHCenter)
        empty_layout.addStretch()
        self.stack.addWidget(self.empty_page)
        self._build_series_page()
        self._build_analysis_page()
        self.set_series(None)

    def _build_series_page(self) -> None:
        layout = QVBoxLayout(self.series_page)
        toolbar = QHBoxLayout()
        self.run_button = QPushButton("Match peaks")
        self.run_button.clicked.connect(self._run)
        self.name_track = QPushButton("Name trend…")
        self.name_track.clicked.connect(self._name_track)
        self.add_to_table = QPushButton("Add trend to table…")
        self.add_to_table.clicked.connect(self._add_to_table)
        self.settings_button = QPushButton("Matching settings…")
        self.settings_button.clicked.connect(self._settings)
        for button in (self.run_button, self.name_track, self.add_to_table, self.settings_button):
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        self.details = QCheckBox("Details")
        self.details.setToolTip("Show model, width and relative area")
        toolbar.addWidget(self.details)
        layout.addLayout(toolbar)
        self.series_tabs = QTabWidget()
        self.series_tabs.setObjectName("PostProcessingViews")
        self.track_list = QListWidget()
        self.track_list.setObjectName("PostProcessingTrendList")
        self.track_list.setMinimumWidth(170)
        self.track_list.setMaximumWidth(280)
        self.track_list.currentItemChanged.connect(self._track_changed)
        self.series_table = AnalysisTableView(trend=True)
        self.details.toggled.connect(self.series_table.set_details)
        self.preview = TrendWidget()
        self.preview.columns.setMaximumWidth(150)
        self.links = LinkEditor(self._controller)
        self.component_splitter = QSplitter()
        self.component_splitter.addWidget(self.track_list)
        self.component_splitter.addWidget(self.series_table)
        self.component_splitter.setStretchFactor(1, 1)
        self.component_splitter.setSizes([190, 800])
        self.components_page = QWidget()
        components_layout = QVBoxLayout(self.components_page)
        components_layout.setContentsMargins(0, 6, 0, 0)
        self.trend_hint = QLabel(
            "Select a trend on the left. Double-click a row to open its decomposition."
        )
        components_layout.addWidget(self.trend_hint)
        components_layout.addWidget(self.component_splitter, 1)
        self.preview_splitter = QSplitter()
        self.preview_splitter.addWidget(self.preview)
        self.series_tabs.addTab(self.components_page, "Trend components")
        self.series_tabs.addTab(self.preview_splitter, "Trend preview")
        self.series_tabs.addTab(self.links, "Neighbouring links")
        layout.addWidget(self.series_tabs, 1)
        self.series_tabs.currentChanged.connect(self._series_view_changed)
        self.series_table.doubleClicked.connect(self._activate_series_row)
        self.preview.sourceActivated.connect(self._activate_preview)

    def _series_view_changed(self) -> None:
        trends_visible = self.series_tabs.currentWidget() is not self.links
        if trends_visible:
            target = (
                self.component_splitter
                if self.series_tabs.currentIndex() == 0
                else self.preview_splitter
            )
            if self.track_list.parentWidget() is not target:
                target.insertWidget(0, self.track_list)
                target.setStretchFactor(1, 1)
                target.setSizes([190, 800])
        self.track_list.setVisible(trends_visible)
        self.name_track.setVisible(trends_visible)
        self.add_to_table.setVisible(trends_visible)
        self.details.setVisible(self.series_tabs.currentIndex() == 0)
        self.status.setText(
            self._series_status
            if trends_visible
            else self._series_status.replace(
                "Select a peak trend, inspect its components, then add it to a table",
                "Compare spectra and edit links between neighbouring steps",
            )
        )

    def _build_analysis_page(self) -> None:
        layout = QVBoxLayout(self.analysis_page)
        toolbar = QHBoxLayout()
        self.metrics = QComboBox()
        self.metrics.setMinimumWidth(160)
        for label, callback in (
            ("Sources…", self._sources),
            ("Update alignment", self._align),
            ("Add column…", self._add_column),
            ("Edit column…", self._edit_column),
            ("Remove column", self._remove_column),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            toolbar.addWidget(button)
        layout.addLayout(toolbar)
        row = QHBoxLayout()
        row.addWidget(self.metrics, 1)
        for label, direction in (("Left", -1), ("Right", 1)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, d=direction: self._move_column(d))
            row.addWidget(button)
        self.precision = QSpinBox()
        self.precision.setRange(-1, 15)
        self.precision.setValue(-1)
        self.precision.setSpecialValueText("Full precision")
        self.precision.setToolTip("CSV decimal places; calculations retain full precision")
        row.addWidget(self.precision)
        self.export_button = QPushButton("Export CSV…")
        self.export_button.clicked.connect(self._export)
        row.addWidget(self.export_button)
        layout.addLayout(row)
        self.source_summary = QLabel()
        self.source_summary.setWordWrap(True)
        layout.addWidget(self.source_summary)
        self.table = AnalysisTableView(sortable=False)
        self.trends = TrendWidget()
        self.trends.selectionChanged.connect(self._plot_selection)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("PostProcessingViews")
        self.tabs.addTab(self.table, "Analytical table")
        self.tabs.addTab(self.trends, "Graphs")
        layout.addWidget(self.tabs, 1)
        self.table.doubleClicked.connect(self._activate_cell)
        self.trends.sourceActivated.connect(self._activate_graph)

    def set_series(self, series_id: str | None) -> None:
        """Open a series without changing the saved table or previous trend selection."""
        self._kind, self.series_id = "series", series_id
        self.stack.setCurrentWidget(self.series_page)
        self.refresh()

    def set_analytical_table(self, table_id: str) -> None:
        """Open an independently saved analytical table."""
        self._kind, self.table_id = "table", table_id
        self.stack.setCurrentWidget(self.analysis_page)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild presentation from immutable query results after a command."""
        if self._kind == "table":
            self._refresh_analysis()
            return
        sid = self.series_id
        exists = sid in self._controller.query.get_post_processing_series_ids()
        self.status.setVisible(exists)
        self.stack.setCurrentWidget(self.series_page if exists else self.empty_page)
        self.settings_button.setEnabled(exists)
        self.run_button.setEnabled(exists)
        self.name_track.setEnabled(exists)
        self.add_to_table.setEnabled(exists)
        self.track_list.blockSignals(True)
        self.track_list.clear()
        if not exists:
            self.status.setText("Create a series, order its spectra, then match peaks.")
            self.track_list.blockSignals(False)
            self.links.set_series(None)
            self.series_table.source.clear()
            self.preview.graphics.ci.clear()
            return
        dto = self._controller.query.get_post_processing_series(sid)
        self.status.setText(
            f"{dto.series.name} — "
            + (
                "Match peaks to create trends"
                if dto.series.result is None
                else "Saved result is stale — recalculate"
                if dto.stale
                else "Select a peak trend, inspect its components, then add it to a table"
            )
        )
        if dto.conflicts:
            self.status.setText(
                self.status.text()
                + "\n"
                + "\n".join(
                    dict.fromkeys(
                        issue.split(":", 1)[0].replace("_", " ").capitalize()
                        for issue in dto.conflicts
                    )
                )
            )
        if dto.series.result is not None:
            for track in dto.series.result.tracks:
                label = (
                    f"{self._controller.query.get_trend_label(sid, track.id_)}\n{len(track.component_ids)} / {len(dto.series.spectrum_ids)} peaks"
                    + (" • ambiguous" if track.ambiguous else "")
                )
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, track.id_)
                item.setForeground(QColor(color_for_component(component_id=track.anchor_id)))
                self.track_list.addItem(item)
                if track.id_ == self._selected_tracks.get(sid):
                    self.track_list.setCurrentItem(item)
        if self.track_list.currentRow() < 0 and self.track_list.count():
            self.track_list.setCurrentRow(0)
        self.track_list.blockSignals(False)
        self.name_track.setEnabled(self.track_list.count() > 0)
        self.add_to_table.setEnabled(dto.exportable and self.track_list.count() > 0)
        self.links.set_series(sid)
        self._track_changed()
        self._series_status = self.status.text()
        self._series_view_changed()

    def _track_changed(self) -> None:
        item = self.track_list.currentItem()
        if item is None or self.series_id is None:
            self.series_table.source.clear()
            self.preview.graphics.ci.clear()
            return
        track_id = item.data(Qt.ItemDataRole.UserRole)
        self._selected_tracks[self.series_id] = track_id
        detailed, preview = self._controller.query.get_trend_tables(self.series_id, track_id)
        self.series_table.source.set_table(detailed)
        self.series_table.resizeColumnsToContents()
        self.preview.set_table(self.series_id + track_id, preview)

    def _refresh_analysis(self) -> None:
        self.status.show()
        if self.table_id not in self._controller.query.get_analytical_table_ids():
            return
        dto = self._controller.query.get_analytical_table(self.table_id)
        self.status.setText(
            dto.state.name
            + (
                " — update source matching and row alignment"
                if dto.stale
                else " — steps paired by series order"
            )
            + (
                "\n"
                + "\n".join(
                    dict.fromkeys(
                        issue.split(":", 1)[0].replace("_", " ").capitalize()
                        for issue in dto.conflicts
                    )
                )
                if dto.conflicts
                else ""
            )
        )
        self.source_summary.setText(
            "\n".join(
                (*self._controller.query.get_analytical_source_labels(self.table_id), *dto.warnings)
            )
        )
        self.export_button.setEnabled(dto.exportable)
        selected = self.metrics.currentData()
        self.metrics.clear()
        for metric in dto.state.metrics:
            self.metrics.addItem(metric.name, metric.id_)
        found = self.metrics.findData(selected)
        if found >= 0:
            self.metrics.setCurrentIndex(found)
        self.table.source.set_table(dto.table)
        self.table.resizeColumnsToContents()
        for index, column in enumerate(dto.table.columns):
            self.table.setColumnWidth(
                index, 48 if column.id_ == "step" else min(self.table.columnWidth(index), 165)
            )
        self.trends.set_table(self.table_id, dto.table, selected_columns=dto.state.plotted_columns)

    def _create_series(self) -> None:
        dialog = SeriesDialog(self._controller, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            sid = self._controller.create_post_processing_series(
                dialog.name_edit.text().strip(), dialog.spectrum_ids()
            )
            self.set_series(sid)

    def _run(self) -> None:
        if self.series_id:
            self._controller.run_post_processing(self.series_id)

    def _name_track(self) -> None:
        track_id = self._selected_tracks.get(self.series_id)
        if not track_id or self.series_id is None:
            return
        series = self._controller.query.get_post_processing_series(self.series_id).series
        if series.result is None:
            return
        track = next(t for t in series.result.tracks if t.id_ == track_id)
        name, accepted = QInputDialog.getText(
            self,
            "Peak trend name",
            "Name (empty clears it)",
            text=dict(series.track_names).get(track.anchor_id, ""),
        )
        if accepted:
            self._controller.name_post_processing_track(self.series_id, track_id, name)

    def _add_to_table(self) -> None:
        track_id = self._selected_tracks.get(self.series_id)
        if not track_id or self.series_id is None:
            return
        tids = self._controller.query.get_analytical_table_ids()
        labels = [
            "New analytical table…",
            *(
                f"{index}. {self._controller.query.get_analytical_table(t).state.name}"
                for index, t in enumerate(tids, 1)
            ),
        ]
        label, accepted = QInputDialog.getItem(
            self, "Add selected trend", "Table", labels, editable=False
        )
        if not accepted:
            return
        if labels.index(label) == 0:
            name, accepted = QInputDialog.getText(self, "New table", "Name", text="New analysis")
            if not accepted:
                return
            tid = self._controller.create_analytical_table(name, ((self.series_id, track_id),))
        else:
            tid = tids[labels.index(label) - 1]
            state = self._controller.query.get_analytical_table(tid).state
            pairs = tuple((s.series_id, s.track_id) for s in state.sources)
            self._controller.edit_analytical_table(
                tid, selections=tuple(dict.fromkeys((*pairs, (self.series_id, track_id))))
            )
        self.tableCreated.emit(tid)

    def _sources(self) -> None:
        if self.table_id is None:
            return
        state = self._controller.query.get_analytical_table(self.table_id).state
        dialog = SourceDialog(
            self._controller,
            self,
            name=state.name,
            selected=tuple((s.series_id, s.track_id) for s in state.sources),
        )
        if dialog.exec():
            self._controller.edit_analytical_table(
                state.id_, name=dialog.name_edit.text(), selections=dialog.selections()
            )

    def _align(self) -> None:
        if self.table_id:
            self._controller.edit_analytical_table(self.table_id, refresh_alignment=True)

    def _add_column(self) -> None:
        if self.table_id:
            MetricDialog(self._controller, self.table_id, self).exec()

    def _edit_column(self) -> None:
        if self.table_id and self.metrics.currentData():
            MetricDialog(
                self._controller, self.table_id, self, metric_id=self.metrics.currentData()
            ).exec()

    def _remove_column(self) -> None:
        if self.table_id and self.metrics.currentData():
            self._controller.remove_analytical_metric(self.table_id, self.metrics.currentData())

    def _move_column(self, offset: int) -> None:
        if not self.table_id:
            return
        state = self._controller.query.get_analytical_table(self.table_id).state
        ids = [
            c.id_
            for c in self._controller.query.get_analytical_table(self.table_id).table.columns
            if c.id_ in {m.id_ for m in state.metrics}
        ]
        current = self.metrics.currentData()
        if current not in ids:
            return
        index = ids.index(current)
        if 0 <= index + offset < len(ids):
            ids[index], ids[index + offset] = ids[index + offset], ids[index]
            self._controller.edit_analytical_table(self.table_id, column_order=tuple(ids))

    def _plot_selection(self, ids: tuple[str, ...]) -> None:
        if self.table_id:
            state = self._controller.query.get_analytical_table(self.table_id).state
            if state.plotted_columns != ids:
                self._controller.edit_analytical_table(self.table_id, plotted_columns=ids)

    def _export(self) -> None:
        if not self.table_id:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export analysis and provenance", "", "CSV (*.csv)"
        )
        if path:
            self._controller.export_analytical_table(
                self.table_id,
                path if path.lower().endswith(".csv") else path + ".csv",
                precision=None if self.precision.value() < 0 else self.precision.value(),
            )

    def _activate_series_row(self, index: QModelIndex) -> None:
        row = self.series_table.source_row(index)
        if row and row.component_id and self.series_id is not None:
            self._controller.set_trend_color_context(self.series_id)
            source = self._controller.query.get_post_processing_source(
                self.series_id, row.spectrum_id, component_id=row.component_id
            )
            if source:
                self.sourceActivated.emit(*source)

    def _activate_preview(self, spectrum_id: str, _track_id: str | None) -> None:
        if self.series_id is None:
            return
        self._controller.set_trend_color_context(self.series_id)
        source = self._controller.query.get_post_processing_source(
            self.series_id, spectrum_id, self._selected_tracks.get(self.series_id)
        )
        if source:
            self.sourceActivated.emit(*source)

    def _activate_cell(self, index: QModelIndex) -> None:
        if self.table.source.table is None:
            return
        source_index = self.table.proxy.mapToSource(index)
        cell = self.table.source.table.rows[source_index.row()].cells[source_index.column()]
        self._open_sources(cell.sources)

    def _activate_graph(self, spectrum_id: str, column_id: str | None) -> None:
        if not self.table_id:
            return
        table = self._controller.query.get_analytical_table(self.table_id).table
        row = next((r for r in table.rows if r.id_ == spectrum_id), None)
        col = next((i for i, c in enumerate(table.columns) if c.id_ == column_id), None)
        if row is not None and col is not None:
            self._open_sources(row.cells[col].sources)

    def _open_sources(self, sources: tuple[SourceLocation, ...]) -> None:
        if not sources:
            return
        chosen = sources[0]
        if len(sources) > 1:
            labels = [
                f"{getattr(self._controller.query.get_metadata(s.spectrum_id), 'name', None) or s.spectrum_id[:6]} / component {s.component_id[:6]}"
                for s in sources
            ]
            label, accepted = QInputDialog.getItem(
                self, "Contributing components", "Open source", labels, editable=False
            )
            if not accepted:
                return
            chosen = sources[labels.index(label)]
        source = (
            self._controller.query.get_post_processing_source(
                chosen.series_id, chosen.spectrum_id, component_id=chosen.component_id
            )
            if chosen.series_id in self._controller.query.get_post_processing_series_ids()
            else None
        )
        if source:
            self._controller.set_trend_color_context(chosen.series_id)
            self.sourceActivated.emit(*source)

    def _settings(self) -> None:
        if self.series_id is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Matching settings")
        layout = QFormLayout(dialog)
        widgets = {}
        for name, value in self._controller.query.get_matching_settings(self.series_id):
            widget = QDoubleSpinBox()
            widget.setDecimals(6)
            widget.setRange(0.000001 if name.endswith("scale") or name == "energy_gate" else 0, 1e6)
            widget.setValue(value)
            widgets[name] = widget
            layout.addRow(name.replace("_", " "), widget)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec():
            self._controller.set_post_processing_settings(
                self.series_id, {name: w.value() for name, w in widgets.items()}
            )
