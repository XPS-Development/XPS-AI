"""
pyqtgraph-based plot area for spectrum visualization.

Displays the selected spectrum with raw data, background, peaks, model,
and optional residuals. Driven by ControllerWrapper selection and signals,
using ViewerDataProvider protocol and tools.evaluation.spectrum_bundle.
"""

from PySide6.QtCore import Qt
import numpy as np
import pyqtgraph as pg

from tools.evaluation import SpectrumEvaluationResult, spectrum_bundle

from .controller import ControllerWrapper


# Curve styling constants
PEN_RAW = pg.mkPen(color="k", width=1)
PEN_BACKGROUND = pg.mkPen(color="k", width=1, style=Qt.PenStyle.DashLine)
PEN_MODEL = pg.mkPen(color="r", width=1.5)
PEN_RESIDUALS = pg.mkPen(color="#808080", width=1)

# Peak colors (cycled per peak)
PEAK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


class PlotAreaWidget(pg.PlotWidget):
    """
    Plot widget displaying the selected spectrum with fitted components.

    Shows raw spectrum, background, peaks, and model curves. Optionally
    overlays residuals with a vertical offset. Refreshes on collection
    or selection changes via the connected controller signals.

    Parameters
    ----------
    controller : ControllerWrapper
        Controller providing selection state and ViewerDataProvider methods.
    parent : QWidget or None, optional
        Parent widget.
    """

    def __init__(
        self,
        controller: ControllerWrapper,
        parent: pg.QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setBackground("w")
        self.showGrid(x=True, y=True, alpha=0.3)

    def refresh(self) -> None:
        """
        Refresh the plot from the current selection.

        Reads selected spectrum from the controller, fetches data via
        get_spectrum_repr and spectrum_bundle, then redraws all curves.
        Clears the plot if no spectrum is selected or on error.
        """
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            self.clear_plot()
            return

        try:
            spectrum, regions = self._controller.get_spectrum_repr(spectrum_id, normalized=False)
            result = spectrum_bundle(spectrum, regions, include_background=True)
        except Exception:
            self.clear_plot()
            return

        self._draw_spectrum(result)

    def clear_plot(self) -> None:
        """Remove all plot items."""
        self.clear()

    def _draw_spectrum(self, result: SpectrumEvaluationResult) -> None:
        """
        Draw all curves from a SpectrumEvaluationResult.

        Parameters
        ----------
        result : SpectrumEvaluationResult
            Evaluated spectrum with regions, backgrounds, peaks, model.
        """
        self.clear()

        # Raw spectrum (full range)
        self.plot(result.x, result.y, pen=PEN_RAW)

        # Per-region curves
        for region in result.regions:
            x = region.x
            bg_y = np.zeros_like(x) if region.background is None else region.background.y

            # Background
            if region.background is not None:
                self.plot(x, bg_y, pen=PEN_BACKGROUND)

            # Peaks (background + peak contribution)
            for idx, peak in enumerate(region.peaks):
                color = PEAK_COLORS[idx % len(PEAK_COLORS)]
                pen = pg.mkPen(color=color, width=1)
                self.plot(x, bg_y + peak.y, pen=pen)
                bg_y = bg_y + peak.y

            # Model
            self.plot(x, region.model, pen=PEN_MODEL)

            # Residuals (overlaid with vertical offset)
            res = region.residuals
            if res.size > 0:
                y_min, y_max = float(np.min(result.y)), float(np.max(result.y))
                res_max = float(np.max(res))
                offset = y_min - res_max - 0.1 * (y_max - y_min)
                self.plot(x, res + offset, pen=PEN_RESIDUALS)
