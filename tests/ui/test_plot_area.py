"""Tests for :class:`ui.plot_area.PlotAreaWidget` refresh behaviour."""

import sys
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from core.evaluation import PlotCurve, SpectrumPlotData
from ui.controller import ControllerWrapper
from ui.plot_area import PlotAreaWidget


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for QWidget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast(QApplication, app)


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
