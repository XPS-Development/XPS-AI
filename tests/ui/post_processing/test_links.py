"""Stacked spectral comparison and stable editing of adjacent links."""

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtGui import QPen
from PySide6.QtWidgets import QApplication
from tests.conftest import seed_hierarchy_metadata

from ui.controller import ControllerWrapper
from ui.post_processing.links import LinkEditor
from ui.post_processing.panel import PostProcessingPanel


@pytest.fixture
def comparison(hierarchy_collection):
    app = QApplication.instance() or QApplication([])
    controller = ControllerWrapper(collection=hierarchy_collection)
    seed_hierarchy_metadata(controller.orchestrator.ctx.metadata)
    for sid in ("s1", "s2", "s3"):
        controller.create_region(sid, region_id=f"r-{sid}")
        for prefix, center in (("p", 0), ("q", 3)):
            controller.create_peak(
                f"r-{sid}",
                "pseudo-voigt",
                {"amp": 2, "cen": center, "sig": 1, "frac": 0.5},
                peak_id=f"{prefix}-{sid}",
            )
    sid = controller.create_post_processing_series("Comparison", ("s1", "s2", "s3"))
    controller.run_post_processing(sid)
    return app, controller, sid


def test_stack_uses_shared_energy_axis_and_preserves_saved_intensities(comparison):
    _, controller, sid = comparison
    original = controller.query.get_spectrum_plot_data("s1").curves[0]
    saved_y = original.y.copy()
    editor = LinkEditor(controller)
    editor.set_series(sid)
    plot = editor.graphics.ci.getItem(0, 0)
    assert len(editor.graphics.ci.items) == 1
    assert plot.getViewBox().state["xInverted"]
    assert plot.getAxis("bottom").labelUnits == "eV"
    raw = plot.listDataItems()[0]
    np.testing.assert_array_equal(raw.xData, original.x)
    expected = (original.y - original.y.min()) / np.ptp(original.y) + 2 * editor.spacing.value()
    np.testing.assert_allclose(raw.yData, expected)
    editor.normalize.setChecked(True)
    editor.scope.setCurrentIndex(1)
    plot = editor.graphics.ci.getItem(0, 0)
    labels = [item.toPlainText() for item in plot.items if isinstance(item, pg.TextItem)]
    assert len(labels) == 3
    np.testing.assert_array_equal(
        controller.query.get_spectrum_plot_data("s1").curves[0].y, saved_y
    )
    editor.deleteLater()


def test_marker_selection_can_switch_pair_in_whole_series(comparison):
    app, controller, sid = comparison
    editor = LinkEditor(controller)
    editor.set_series(sid)
    editor.scope.setCurrentIndex(1)
    editor.show_links.setChecked(True)
    plot = editor.graphics.ci.getItem(0, 0)
    markers = [item for item in plot.items if isinstance(item, pg.ScatterPlotItem)]
    # Last marker belongs to the second peak in the last spectrum.
    markers[-1].sigClicked.emit(markers[-1], [], None)
    app.processEvents()
    assert editor.pair.currentData() == ("s2", "s3")
    assert editor.right.currentData() == "q-s3"
    editor._select_peak("s2", "q-s2")
    editor._apply(False)
    result = controller.query.get_post_processing_series(sid).series.result
    assert not any(
        link.source_id == "q-s2" and link.target_id == "q-s3"
        for track in result.tracks
        for link in track.links
    )
    controller.undo()
    editor.set_series(sid)
    result = controller.query.get_post_processing_series(sid).series.result
    assert any(
        link.source_id == "q-s2" and link.target_id == "q-s3"
        for track in result.tracks
        for link in track.links
    )
    editor.deleteLater()


def test_table_trend_selection_is_independent_from_link_editor(comparison):
    _, controller, sid = comparison
    panel = PostProcessingPanel(controller)
    panel.set_series(sid)
    panel.links.left.setCurrentIndex(1)
    panel.links.right.setCurrentIndex(0)
    choices = (panel.links.left.currentData(), panel.links.right.currentData())
    plot = panel.links._plot
    panel.track_list.setCurrentRow(1)
    assert (panel.links.left.currentData(), panel.links.right.currentData()) == choices
    assert panel.links._plot is plot
    panel.series_tabs.setCurrentWidget(panel.links)
    assert panel.track_list.isHidden()
    assert panel.name_track.isHidden()
    assert panel.add_to_table.isHidden()
    panel.series_tabs.setCurrentWidget(panel.series_table)
    assert not panel.track_list.isHidden()
    assert panel.track_list.currentRow() == 1
    panel.deleteLater()


def test_waterfall_overlap_is_adjustable_and_overlays_are_optional(comparison):
    _, controller, sid = comparison
    editor = LinkEditor(controller)
    editor.set_series(sid)
    assert editor.scope.currentText() == "Whole series"
    plot = editor.graphics.ci.getItem(0, 0)
    layers = [item for item in plot.listDataItems() if item.opts.get("fillLevel") is not None]
    assert len(layers) == 3
    # Nearer spectra mask the lower part of spectra behind them.
    assert layers[0].zValue() < layers[1].zValue() < layers[2].zValue()
    assert layers[0].opts["fillLevel"] == pytest.approx(0.36)
    assert layers[2].opts["fillLevel"] == 0
    assert not any(isinstance(item, pg.ScatterPlotItem) and item.isVisible() for item in plot.items)
    assert len([item for item in plot.listDataItems() if item.isVisible()]) == 3
    original = layers[0].yData.copy()
    editor.spacing.setValue(0.1)
    plot = editor.graphics.ci.getItem(0, 0)
    np.testing.assert_allclose(plot.listDataItems()[0].yData, original - 0.16)
    editor.components.setChecked(True)
    plot = editor.graphics.ci.getItem(0, 0)
    assert len([item for item in plot.listDataItems() if item.isVisible()]) > 3
    editor.show_links.setChecked(True)
    plot = editor.graphics.ci.getItem(0, 0)
    assert any(isinstance(item, pg.ScatterPlotItem) for item in plot.items)
    editor.deleteLater()


def test_pair_and_peak_selection_reuse_curves_and_preserve_zoom(comparison, monkeypatch):
    _, controller, sid = comparison
    editor = LinkEditor(controller)
    editor.set_series(sid)
    editor.components.setChecked(True)
    editor.show_links.setChecked(True)
    plot = editor._plot
    assert plot is not None
    curves = tuple(plot.listDataItems())
    view = plot.getViewBox()
    view.setXRange(-2, 4, padding=0)
    view.setYRange(0.2, 1.2, padding=0)
    ranges = view.viewRange()

    def unexpected_evaluation(*_args):
        pytest.fail("Selection must not evaluate spectrum curves again")

    monkeypatch.setattr(controller.query, "get_spectrum_plot_data", unexpected_evaluation)
    editor.pair.setCurrentIndex(1)
    editor.left.setCurrentIndex(1)
    editor.right.setCurrentIndex(0)
    editor.set_series(sid)
    editor.components.setChecked(False)
    editor.show_links.setChecked(False)
    editor.show_links.setChecked(True)
    assert editor._plot is plot
    assert tuple(plot.listDataItems()) == curves
    np.testing.assert_allclose(view.viewRange(), ranges)
    for _, _, line, color in editor._link_items:
        assert color.alpha() == 255
        pen = line.opts["pen"]
        assert isinstance(pen, QPen)
        assert pen.widthF() >= 2
    editor.deleteLater()


def test_changed_inputs_rebuild_curves_but_pair_mode_only_rebuilds_when_needed(comparison):
    _, controller, sid = comparison
    editor = LinkEditor(controller)
    editor.set_series(sid)
    initial = editor._plot
    editor.scope.setCurrentIndex(0)
    pair_plot = editor._plot
    assert pair_plot is not initial
    editor.left.setCurrentIndex(1)
    assert editor._plot is pair_plot
    editor.set_series(sid)
    assert editor._plot is pair_plot
    editor.pair.setCurrentIndex(1)
    assert editor._plot is not pair_plot
    old = editor._plot
    controller.update_parameter("p-s2", "amp", "value", 8)
    editor.set_series(sid)
    assert editor._plot is not old
    editor.deleteLater()


def test_hovering_peak_shows_energy_tooltip_and_clears_it_on_exit(comparison):
    app, controller, sid = comparison
    editor = LinkEditor(controller)
    editor.set_series(sid)
    editor.show_links.setChecked(True)
    editor.show()
    app.processEvents()
    marker = next(iter(editor._markers.values()))
    point = marker.points()[0]
    marker.hoverEvent(SimpleNamespace(exit=False, pos=point.pos))
    view = marker.getViewBox()
    assert view is not None
    assert view.toolTip() == point.data()
    assert "eV" in view.toolTip()
    marker.hoverEvent(SimpleNamespace(exit=True))
    assert view.toolTip() == ""
    editor.close()
