"""Tests for :class:`app.query_service.QueryService`."""

from app.query_service import QueryService
from core.services import CoreContext


def test_get_spectrum_plot_data_returns_display_curves(simple_collection, spectrum_id: str) -> None:
    """get_spectrum_plot_data evaluates the spectrum and returns plot-ready curves."""
    ctx = CoreContext.from_collection(simple_collection)
    query = QueryService(ctx)

    plot_data = query.get_spectrum_plot_data(spectrum_id)

    kinds = {curve.kind for curve in plot_data.curves}
    assert "raw" in kinds
    assert "model" in kinds
    assert plot_data.residual_y_range is not None
