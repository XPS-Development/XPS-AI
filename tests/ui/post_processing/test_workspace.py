import sys
from typing import cast

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from tests.conftest import seed_hierarchy_metadata

from core.dto import AnalysisCell, AnalysisColumn, AnalysisRow, AnalysisTable
from ui.controller import ControllerWrapper
from ui.main_window import MainWindow
from ui.post_processing.metric_dialog import MetricDialog
from ui.post_processing.series_dialog import SeriesDialog
from ui.post_processing.table import AnalysisTableView
from ui.post_processing.trends import TrendWidget


@pytest.fixture(scope="module")
def qapp():
    return cast(QApplication, QApplication.instance() or QApplication(sys.argv))


def test_selection_of_multiple_branches_search_and_independent_order(qapp, hierarchy_collection):
    controller = ControllerWrapper(collection=hierarchy_collection)
    seed_hierarchy_metadata(controller.orchestrator.ctx.metadata)
    dialog = SeriesDialog(controller)
    file = dialog.tree.topLevelItem(0)
    assert file is not None
    file.setCheckState(0, Qt.CheckState.Checked)
    assert dialog.spectrum_ids() == ("s1", "s2", "s3")
    dialog.ordered.setCurrentRow(2)
    dialog._move(-1)
    assert dialog.spectrum_ids() == ("s1", "s3", "s2")
    dialog.search.setText("spec-3")
    assert dialog._leaves["s1"].isHidden()
    assert not dialog._leaves["s3"].isHidden()
    assert dialog.spectrum_ids() == ("s1", "s3", "s2")
    dialog._remove()
    assert dialog.spectrum_ids() == ("s1", "s2")
    dialog._leaves["s3"].setCheckState(0, Qt.CheckState.Checked)
    assert dialog.spectrum_ids() == ("s1", "s2", "s3")
    dialog.deleteLater()


def test_edit_selection_preserves_missing_steps_and_explicit_initial_order(
    qapp, hierarchy_collection
):
    controller = ControllerWrapper(collection=hierarchy_collection)
    seed_hierarchy_metadata(controller.orchestrator.ctx.metadata)
    dialog = SeriesDialog(controller, spectrum_ids=("s3", "deleted", "s1"))
    assert dialog.spectrum_ids() == ("s3", "deleted", "s1")
    dialog._leaves["s2"].setCheckState(0, Qt.CheckState.Checked)
    assert dialog.spectrum_ids() == ("s3", "deleted", "s1", "s2")
    dialog.ordered.setCurrentRow(1)
    dialog._remove()
    assert dialog.spectrum_ids() == ("s3", "s1", "s2")
    dialog.deleteLater()


def test_numeric_sorting_preserves_navigation_and_table_source_order(qapp):
    table = AnalysisTable(
        (AnalysisColumn("step", "Step"), AnalysisColumn("area", "Area")),
        (
            AnalysisRow(1, "first", (AnalysisCell(1), AnalysisCell(10))),
            AnalysisRow(2, "second", (AnalysisCell(2), AnalysisCell(2))),
        ),
    )
    view = AnalysisTableView()
    view.source.set_table(table)
    view.sortByColumn(1, Qt.SortOrder.AscendingOrder)
    row = view.source_row(view.proxy.index(0, 1))
    assert row is not None
    assert row.spectrum_id == "second"
    assert [row.spectrum_id for row in table.rows] == ["first", "second"]
    view.deleteLater()


def test_trends_keep_gaps_and_separate_units(qapp):
    table = AnalysisTable(
        (
            AnalysisColumn("step", "Step"),
            AnalysisColumn("spectrum", "Spectrum"),
            AnalysisColumn("energy", "Energy", "eV"),
            AnalysisColumn("area", "Area", "area"),
        ),
        (
            AnalysisRow(
                1, "s1", (AnalysisCell(1), AnalysisCell("S1"), AnalysisCell(1), AnalysisCell(5))
            ),
            AnalysisRow(
                2,
                "s2",
                (
                    AnalysisCell(2),
                    AnalysisCell("S2"),
                    AnalysisCell(None, "missing", "missing_peak"),
                    AnalysisCell(None, "missing", "missing_peak"),
                ),
            ),
            AnalysisRow(
                3,
                "s3",
                (
                    AnalysisCell(3),
                    AnalysisCell("S3"),
                    AnalysisCell(2, "ambiguous"),
                    AnalysisCell(9),
                ),
            ),
        ),
    )
    widget = TrendWidget()
    widget.set_table("series", table)
    plots = list(widget.graphics.ci.items)
    assert len(plots) == 2
    lines = [
        item for plot in plots for item in plot.listDataItems() if isinstance(item, pg.PlotDataItem)
    ]
    assert len(lines) == 2
    assert np.isnan(lines[0].yData[1])
    assert list(lines[0].xData) == [1, 2, 3]
    assert lines[0].opts["connect"] == "finite"
    widget.columns.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert len(widget.graphics.ci.items) == 1
    widget.set_table("series", table)
    assert widget.columns.item(0).checkState() == Qt.CheckState.Unchecked
    widget.deleteLater()


def test_controller_emits_post_processing_changes_on_run_undo_and_fit(qapp, simple_collection):
    controller = ControllerWrapper(collection=simple_collection)
    signals = []
    controller.postProcessingChanged.connect(lambda: signals.append(True))
    sid = controller.create_post_processing_series("S", ["s1"])
    controller.run_post_processing(sid)
    controller.undo()
    controller.redo()
    controller.update_parameter("p1", "amp", "value", 2)
    assert len(signals) == 5


def test_metric_dialog_preserves_explicit_reference_and_operand_selection(
    qapp, hierarchy_collection
):
    controller = ControllerWrapper(collection=hierarchy_collection)
    seed_hierarchy_metadata(controller.orchestrator.ctx.metadata)
    sid = controller.create_post_processing_series("S", ["s1", "s2"])
    controller.run_post_processing(sid)
    # Empty spectra have no operands; the dialog must not fabricate a track.
    tid = controller.create_analytical_table("Empty")
    dialog = MetricDialog(controller, tid)
    dialog.operation.setCurrentIndex(dialog.operation.findData("position_shift"))
    assert dialog.reference.currentData() is None
    dialog._save()
    assert dialog.result() == 0
    assert "operands" in dialog.error.text()
    dialog.deleteLater()


def test_mode_retains_workspace_and_source_navigation_reuses_decomposition(
    qapp, simple_collection, monkeypatch
):
    controller = ControllerWrapper(collection=simple_collection)
    sid = controller.create_post_processing_series("S", ["s1"])
    controller.run_post_processing(sid)
    window = MainWindow(controller)
    window.show()
    QApplication.processEvents()
    panel = window._post_processing_panel
    window._mode_tabs.setCurrentIndex(1)
    panel.series_tabs.setCurrentIndex(1)
    window._mode_tabs.setCurrentIndex(0)
    window._mode_tabs.setCurrentIndex(1)
    assert panel.series_id == sid
    assert panel.series_tabs.currentIndex() == 1
    assert window._properties_panel is not None
    assert window._properties_panel.isHidden()
    panel.series_tabs.setCurrentIndex(0)
    panel._activate_series_row(panel.series_table.proxy.index(0, 0))
    QApplication.processEvents()
    assert window._mode_tabs.currentIndex() == 0
    assert controller.selected_component_id == "p1"
    assert controller.selected_region_id == "r1"
    assert window._plot_area is not None
    assert window._plot_area._last_spectrum_id == "s1"
    window._mode_tabs.setCurrentIndex(1)
    controller.update_parameter("p1", "amp", "value", 2)
    QApplication.processEvents()
    assert not panel.add_to_table.isEnabled()
    controller.undo()
    QApplication.processEvents()
    assert panel.add_to_table.isEnabled()
    monkeypatch.setattr(window, "_confirm_close", lambda: True)
    window.close()
    window.deleteLater()
    QApplication.processEvents()


def test_metadata_refresh_updates_post_processing_labels_without_invalidating_fit(
    qapp, simple_collection
):
    controller = ControllerWrapper(collection=simple_collection)
    sid = controller.create_post_processing_series("S", ["s1"])
    controller.run_post_processing(sid)
    signals = []
    controller.postProcessingChanged.connect(lambda: signals.append(True))
    controller.rename_spectrum("s1", "New name")
    assert len(signals) == 1
    assert controller.query.get_post_processing_table(sid).rows[0].cells[1].value == "New name"
    assert not controller.query.get_post_processing_series(sid).stale


def test_copy_selection_preserves_zero_and_empty_cells(qapp):
    from PySide6.QtCore import QItemSelectionModel
    from PySide6.QtTest import QTest

    table = AnalysisTable(
        (AnalysisColumn("zero", "Zero"), AnalysisColumn("empty", "Empty")),
        (AnalysisRow(1, "s", (AnalysisCell(0.0), AnalysisCell(None, "missing", "missing_peak"))),),
    )
    view = AnalysisTableView()
    view.source.set_table(table)
    for column in (0, 1):
        view.selectionModel().select(
            view.proxy.index(0, column), QItemSelectionModel.SelectionFlag.Select
        )
    QTest.keyClick(view, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert QApplication.clipboard().text() == "0\t"
    view.deleteLater()


def test_cross_series_source_dialog_and_sum_ratio_editor(qapp, simple_collection):
    from ui.post_processing.source_dialog import SourceDialog

    controller = ControllerWrapper(collection=simple_collection)
    first = controller.create_post_processing_series("Ag", ["s1"])
    second = controller.create_post_processing_series("C", ["s1"])
    controller.run_post_processing(first)
    controller.run_post_processing(second)
    tid = controller.create_analytical_table(
        "Analysis", ((first, "track/p1"), (second, "track/p1"))
    )
    sources = SourceDialog(controller, selected=((first, "track/p1"), (second, "track/p1")))
    assert set(sources.selections()) == {(first, "track/p1"), (second, "track/p1")}
    root = sources.tree.topLevelItem(0)
    assert root is not None
    item = root.child(0)
    assert item is not None
    assert "E:" in item.text(2)
    dialog = MetricDialog(controller, tid)
    dialog.name_edit.setText("Ag / C")
    dialog.operation.setCurrentIndex(dialog.operation.findData("area_sum_ratio"))
    dialog.numerator.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.denominator.item(1).setCheckState(Qt.CheckState.Checked)
    dialog._save()
    metric = controller.query.get_analytical_table(tid).state.metrics[-1]
    assert metric.name == "Ag / C"
    assert metric.tracks == (controller.query.get_analytical_table(tid).state.sources[0].id_,)
    assert len(metric.denominator) == 1
    assert controller.query.get_analytical_table(tid).table.rows[0].cells[-1].value == 1
    sources.deleteLater()
    dialog.deleteLater()


def test_independent_table_navigation_and_saved_graph_selection(
    qapp, simple_collection, monkeypatch
):
    controller = ControllerWrapper(collection=simple_collection)
    sid = controller.create_post_processing_series("Ag", ["s1"])
    controller.run_post_processing(sid)
    tid = controller.create_analytical_table("Analysis", ((sid, "track/p1"),))
    window = MainWindow(controller)
    window.show()
    window._mode_tabs.setCurrentIndex(1)
    window._series_panel.select_table(tid)
    QApplication.processEvents()
    panel = window._post_processing_panel
    assert panel.table_id == tid
    assert window._series_panel.tables.count() == 1
    assert panel.table.isColumnHidden(1)
    assert panel.table.source.table is not None
    assert panel.table.source.table.rows[0].cells[-1].sources[0].component_id == "p1"
    panel.trends.columns.item(0).setCheckState(Qt.CheckState.Unchecked)
    QApplication.processEvents()
    assert controller.query.get_analytical_table(tid).state.plotted_columns == ()
    controller.undo()
    QApplication.processEvents()
    assert panel.trends.columns.item(0).checkState() == Qt.CheckState.Checked
    panel._activate_cell(panel.table.proxy.index(0, 3))
    QApplication.processEvents()
    assert controller.selected_component_id == "p1"
    assert window._mode_tabs.currentIndex() == 0
    window._mode_tabs.setCurrentIndex(1)
    assert panel.table_id == tid
    monkeypatch.setattr(window, "_confirm_close", lambda: True)
    window.close()
    window.deleteLater()
