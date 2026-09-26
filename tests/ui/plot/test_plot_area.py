"""Tests for :class:`ui.plot.plot_area.PlotAreaWidget` refresh behaviour."""

import sys
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from core.evaluation import PlotCurve, SpectrumPlotData
from ui.controller import ControllerWrapper
from ui.plot.plot_area import PlotAreaWidget


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for QWidget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast(QApplication, app)


def test_refresh_inverts_x_axis_from_parameters(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
) -> None:
    """The X axis follows ``AppParameters.invert_x_axis`` on both plots."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.set_selection(spectrum_id)
    widget = PlotAreaWidget(controller)

    controller.orchestrator.params.invert_x_axis = True
    widget.refresh()
    assert widget._main_plot.getViewBox().xInverted() is True
    assert widget._res_plot.getViewBox().xInverted() is True

    controller.orchestrator.params.invert_x_axis = False
    widget.refresh()
    assert widget._main_plot.getViewBox().xInverted() is False
    assert widget._res_plot.getViewBox().xInverted() is False


def test_plot_area_refresh_uses_query_service(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
) -> None:
    """refresh() loads display data from the controller query API."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    plot_data = SpectrumPlotData(
        curves=(
            PlotCurve(
                x=np.array([0.0, 1.0]),
                y=np.array([1.0, 2.0]),
                kind="raw",
            ),
        ),
        residual_y_range=(-1.0, 1.0),
    )
    controller.query.get_spectrum_plot_data = MagicMock(return_value=plot_data)
    controller.set_selection(spectrum_id)
    widget = PlotAreaWidget(controller)

    widget.refresh()

    controller.query.get_spectrum_plot_data.assert_called_once_with(
        spectrum_id,
        normalized=False,
    )
    assert widget._last_plot_data is plot_data
    first_items = list(widget._main_plot.listDataItems())

    updated = SpectrumPlotData(
        curves=(
            PlotCurve(
                x=np.array([0.0, 1.0]),
                y=np.array([3.0, 4.0]),
                kind="raw",
            ),
        ),
        residual_y_range=(-1.0, 1.0),
    )
    controller.query.get_spectrum_plot_data = MagicMock(return_value=updated)
    widget.refresh()

    second_items = list(widget._main_plot.listDataItems())
    assert second_items == first_items
    _x, y = second_items[0].getData()
    assert np.allclose(y, [3.0, 4.0])


def test_overlay_labels_stay_inside_the_data_area(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
) -> None:
    """χ² stays off the axis gutter, and cursor text stays inside the plot."""
    del spectrum_id
    controller = ControllerWrapper(collection=simple_collection)
    widget = PlotAreaWidget(controller)
    widget.resize(640, 480)
    widget.show()
    qapp.processEvents()

    widget._chi_label.setText("χ² = 465.158    χ²/dof = 1.679")
    widget._chi_label.show()
    widget._position_chi_label()
    chi_rect = widget._viewbox_rect(widget._res_plot)
    assert widget._chi_label.x() >= chi_rect.left()
    assert widget._chi_label.x() + widget._chi_label.width() <= chi_rect.right() + 1

    cursor = widget._cursor_label
    assert cursor is not None
    cursor.setText("x: 301.6  y: 123456.7")
    widget._position_cursor_label()
    data_rect = widget._viewbox_rect(widget._main_plot)
    assert cursor.x() >= data_rect.left()
    assert cursor.x() + cursor.width() <= data_rect.right() + 1
    widget.close()


def test_plot_area_refresh_clears_when_no_selection(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
) -> None:
    """refresh() clears the plot when no spectrum is selected."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.query.get_spectrum_plot_data = MagicMock()
    controller.set_selection(spectrum_id)
    widget = PlotAreaWidget(controller)
    widget.refresh()

    controller.set_selection(None)
    widget.refresh()

    controller.query.get_spectrum_plot_data.assert_called_once()
    assert widget._last_plot_data is None


def test_plot_area_refresh_handles_stale_selection(
    qapp: QApplication,
    simple_collection,
    spectrum_id: str,
) -> None:
    """refresh() clears the plot if the selected spectrum was already removed."""
    del qapp
    controller = ControllerWrapper(collection=simple_collection)
    controller.set_selection(spectrum_id)
    widget = PlotAreaWidget(controller)
    widget.refresh()
    assert widget._last_plot_data is not None

    # Simulate a deleted spectrum with a stale controller selection.
    controller._selected_spectrum_id = spectrum_id
    controller.orchestrator.full_remove_object(spectrum_id)
    widget.refresh()

    assert controller.selected_spectrum_id is None
    assert widget._last_plot_data is None
