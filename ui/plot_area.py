"""
pyqtgraph-based plot area for spectrum visualization.

Displays the selected spectrum with raw data, background, peaks, model,
and optional residuals. Driven by ControllerWrapper selection and signals,
using ViewerDataProvider protocol and tools.evaluation.spectrum_bundle.
"""

from collections.abc import Iterable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QMenu, QVBoxLayout, QWidget

from tools.evaluation import SpectrumEvaluationResult, spectrum_bundle

from .context_menus import (
    SpectrumContextMenuActions,
    attach_region_context_actions,
    attach_spectrum_context_actions,
)
from .controller import ControllerWrapper


# Curve styling constants
PEN_RAW = pg.mkPen(color="k", width=1)
PEN_BACKGROUND = pg.mkPen(color="k", width=1, style=Qt.PenStyle.DashLine)
PEN_MODEL = pg.mkPen(color="r", width=1.5)
PEN_RESIDUALS = pg.mkPen(color="#808080", width=2)

REGION_BOUNDS_PEN = pg.mkPen(color="#000000", width=3)
REGION_BOUNDS_HOVER_PEN = pg.mkPen(color="#000000", width=4)
REGION_BOUNDS_BRUSH = pg.mkBrush(0, 0, 0, 0)
REGION_BOUNDS_HOVER_BRUSH = pg.mkBrush(0, 0, 255, 10)

# Peak colors (cycled per peak)
PEAK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


class VieBoxCustomContextMenu(pg.ViewBox):
    """
    ViewBox with a custom context menu.
    """

    def __init__(
        self,
        *,
        controller: ControllerWrapper,
        **kwargs,
    ) -> None:
        self._controller = controller
        self._spectrum_menu_actions: SpectrumContextMenuActions | None = None

        super().__init__(**kwargs)

    def _create_menu(self) -> QMenu:
        menu = QMenu()
        self._spectrum_menu_actions = attach_spectrum_context_actions(
            menu, self._controller, None, include_model_actions=False
        )
        return menu

    def _applyMenuEnabled(self):
        enableMenu = self.state.get("enableMenu", True)

        if enableMenu and self.menu is None:
            self.menu = self._create_menu()

        elif not enableMenu and self.menu is not None:
            self.menu.setParent(None)
            self.menu = None
            self._spectrum_menu_actions = None

    def _update_menu_enabled_state(self) -> None:
        """Enable/disable spectrum-level actions based on current selection."""
        if self._spectrum_menu_actions is not None:
            self._spectrum_menu_actions.update_enabled_state()

    def raiseContextMenu(self, ev):
        """Raise the context menu for the view box without attaching scene menus."""
        menu = self.getMenu(ev)
        if menu is not None:
            self._update_menu_enabled_state()
            menu.popup(ev.screenPos().toPoint())


class RegionContextPlotWidget(pg.PlotWidget):
    """
    Plot widget with a region-aware context menu.

    The context menu changes depending on whether the click occurs inside an ROI:

    - Inside an ROI: select that region and show region actions.
    - Outside all ROIs: show spectrum-level actions.
    """

    def __init__(
        self,
        *,
        controller: ControllerWrapper,
        parent: QWidget | None = None,
    ) -> None:
        self.menu = None
        self._controller = controller

        # create view box with custom context menu and disable all default menus
        super().__init__(
            parent=parent,
            viewBox=VieBoxCustomContextMenu(
                controller=controller,
                enableMenu=False,
            ),
            enableMenu=False,
        )

        vb = self.plotItem.getViewBox()
        vb.setMenuEnabled(True)


class InteractiveRegion(pg.LinearRegionItem):
    """
    LinearRegionItem carrying a region_id and emitting a click signal.

    Parameters
    ----------
    region_id : str
        Identifier of the attached region.
    values : tuple[float, float]
        Initial region bounds in x-axis values.
    """

    sigClickedRegion: Signal = Signal(str)

    def __init__(
        self,
        region_id: str,
        values: tuple[float, float],
        *,
        controller: ControllerWrapper,
        dialog_parent: QWidget,
        **kwargs,
    ) -> None:
        super().__init__(values=values, **kwargs)
        self.region_id = region_id
        self._controller = controller
        self._dialog_parent = dialog_parent

        self.menu = QMenu()
        self._region_menu_actions = attach_region_context_actions(
            self.menu,
            controller,
            region_id,
            dialog_parent,
            include_model_actions=False,
        )
        self._update_menu_enabled_state()

    def mouseClickEvent(self, ev) -> None:  # noqa: ANN001
        if ev.button() == Qt.MouseButton.LeftButton or ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self.sigClickedRegion.emit(self.region_id)
            if ev.button() == Qt.MouseButton.RightButton:
                self.raiseContextMenu(ev)

    def mouseDragEvent(self, ev):
        """Ignore mouse drag events."""
        pass

    def _update_menu_enabled_state(self) -> None:
        """Enable/disable ROI actions based on current region state."""
        self._region_menu_actions.update_enabled_state()

    def raiseContextMenu(self, ev):
        self._update_menu_enabled_state()
        self.menu.popup(ev.screenPos().toPoint())
        return True


class PlotAreaWidget(QWidget):
    """
    Plot widget displaying the selected spectrum with fitted components.

    Shows main plot (raw spectrum, background, peaks, model), a separate
    residuals subplot with shared x-axis and locked y-axis, optional ROI
    for the selected region, cursor (x, y) overlay, and context menu for
    region-aware actions. Refreshes on collection or selection changes via
    the connected controller signals.

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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        pg.setConfigOption("leftButtonPan", True)
        self._controller = controller
        self._roi_items_by_region: dict[str, InteractiveRegion] = {}
        self._roi_region_ids_in_plot: set[str] = set()
        self._cursor_label: QLabel | None = None
        self._last_result: SpectrumEvaluationResult | None = None
        self._last_spectrum_id: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main plot (spectrum, background, peaks, model)
        self._main_plot = RegionContextPlotWidget(controller=self._controller)
        self._main_plot.setBackground("w")
        self._main_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._main_plot, stretch=1)

        # Residuals plot (shared x-axis, locked y)
        self._res_plot = pg.PlotWidget(parent=self, enableMenu=False)
        self._res_plot.setBackground("w")
        self._res_plot.showGrid(x=True, y=True, alpha=0.3)
        self._res_plot.setMinimumHeight(80)
        self._res_plot.getViewBox().setXLink(self._main_plot.getViewBox())
        self._res_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        layout.addWidget(self._res_plot, stretch=0)

        # Cursor (x, y) overlay on main plot
        self._cursor_label = QLabel(self._main_plot)
        self._cursor_label.setStyleSheet(
            "background-color: rgba(255,255,255,0.8); padding: 2px 4px; border-radius: 2px;"
        )
        self._cursor_label.setText("x: —  y: —")
        self._cursor_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._main_plot.scene().sigMouseMoved.connect(self._on_main_plot_mouse_moved)

    def _on_main_plot_mouse_moved(self, pos: pg.QtCore.QPointF) -> None:
        """Update cursor (x, y) label from main plot mouse position."""
        if self._cursor_label is None:
            return
        vb = self._main_plot.getViewBox()
        # pos is in scene coordinates
        if vb.sceneBoundingRect().contains(pos):
            coord = vb.mapSceneToView(pos)
            self._cursor_label.setText(f"x: {coord.x():.4g}  y: {coord.y():.4g}")
        else:
            self._cursor_label.setText("x: —  y: —")

    def resizeEvent(self, event) -> None:
        """Reposition cursor label at upper-right of main plot."""
        super().resizeEvent(event)
        self._position_cursor_label()

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

        spectrum, regions = self._controller.get_spectrum_repr(spectrum_id, normalized=False)
        result = spectrum_bundle(spectrum, regions, include_background=True)

        self._last_result = result
        self._sync_residuals_visibility()
        self._sync_rois_for_spectrum(spectrum_id=spectrum_id)
        self._draw_spectrum(result)
        self._position_cursor_label()

    def _position_cursor_label(self) -> None:
        """Place cursor label at upper-right of main plot."""
        if self._cursor_label is not None:
            self._cursor_label.adjustSize()
            self._cursor_label.move(self._main_plot.width() - self._cursor_label.width() - 8, 8)

    def clear_plot(self) -> None:
        """Remove all plot items and ROIs."""
        self._main_plot.clear()
        self._res_plot.clear()
        self._clear_rois()
        self._last_result = None
        self._last_spectrum_id = None

    def _clear_rois(self) -> None:
        """Remove all ROIs from the main plot and forget them."""
        for roi in self._roi_items_by_region.values():
            try:
                self._main_plot.removeItem(roi)
            except Exception:
                pass
        self._roi_items_by_region.clear()
        self._roi_region_ids_in_plot.clear()

    def _sync_residuals_visibility(self) -> None:
        """Show/hide (collapse) residuals plot according to app parameters."""
        params = self._controller.get_app_parameters()
        self._res_plot.setVisible(bool(getattr(params, "show_residuals_plot", True)))

    def _sync_rois_for_spectrum(self, *, spectrum_id: str) -> None:
        """
        Ensure ROIs exist for all regions of the selected spectrum.

        ROIs are created/updated from region slices in value mode and are removed when
        regions disappear.
        """
        if self._last_spectrum_id != spectrum_id:
            # New spectrum selection: forget all ROIs and rebuild.
            self._clear_rois()
            self._last_spectrum_id = spectrum_id

        region_ids = list(self._controller.query.get_regions_ids(spectrum_id))
        wanted = set(region_ids)

        # Remove ROIs for deleted regions.
        for rid in list(self._roi_items_by_region.keys()):
            if rid not in wanted:
                roi = self._roi_items_by_region.pop(rid)
                self._roi_region_ids_in_plot.discard(rid)
                try:
                    self._main_plot.removeItem(roi)
                except Exception:
                    pass

        # Create/update ROIs for current regions.
        for rid in region_ids:
            start_val, stop_val = self._controller.query.get_region_slice(rid, mode="value")
            roi = self._roi_items_by_region.get(rid)
            if roi is None:
                roi = self._create_region_roi(region_id=rid, start=float(start_val), stop=float(stop_val))
                self._roi_items_by_region[rid] = roi
                self._roi_region_ids_in_plot.discard(rid)
            else:
                roi.blockSignals(True)
                roi.setRegion((float(start_val), float(stop_val)))
                roi.blockSignals(False)

    def _create_region_roi(self, *, region_id: str, start: float, stop: float) -> InteractiveRegion:
        """Create a styled ROI bound to a region id."""
        roi = InteractiveRegion(
            region_id,
            values=(start, stop),
            movable=True,
            swapMode="block",
            pen=REGION_BOUNDS_PEN,
            hoverPen=REGION_BOUNDS_HOVER_PEN,
            brush=REGION_BOUNDS_BRUSH,
            hoverBrush=REGION_BOUNDS_HOVER_BRUSH,
            controller=self._controller,
            dialog_parent=self,
        )
        roi.sigRegionChangeFinished.connect(lambda _roi=roi: self._on_roi_region_change_finished(_roi))
        roi.sigClickedRegion.connect(self._on_roi_clicked)
        return roi

    def _on_roi_clicked(self, region_id: str) -> None:
        """Select region when clicking its ROI."""
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return
        self._controller.set_selection(spectrum_id, region_id)

    def _iter_rois(self) -> Iterable[InteractiveRegion]:
        """
        Return the current ROI items.

        Returns
        -------
        Iterable[RegionLinearROI]
            Current ROI items attached to regions.
        """
        return self._roi_items_by_region.values()

    def _on_roi_region_change_finished(self, roi: InteractiveRegion) -> None:
        """Apply ROI bounds to region slice (value mode)."""
        low, high = roi.getRegion()
        try:
            self._controller.update_region_slice(roi.region_id, low, high, mode="value")
        except Exception:
            return

    def _draw_spectrum(self, result: SpectrumEvaluationResult) -> None:
        """
        Draw all curves from a SpectrumEvaluationResult.

        Main plot: raw spectrum, per-region background, peaks, model (no residuals).
        Residuals plot: per-region residuals with shared x and locked y range.
        """
        self._main_plot.clear()
        self._res_plot.clear()

        # Re-add ROIs after clearing to keep them visible, but only once per redraw.
        self._roi_region_ids_in_plot.clear()
        for rid, roi in self._roi_items_by_region.items():
            self._main_plot.addItem(roi)
            self._roi_region_ids_in_plot.add(rid)

        # Raw spectrum (full range)
        self._main_plot.plot(result.x, result.y, pen=PEN_RAW)

        all_res_x: list[np.ndarray] = []
        all_res_y: list[np.ndarray] = []

        for region in result.regions:
            x = region.x
            bg_y = np.zeros_like(x) if region.background is None else region.background.y

            # Background
            if region.background is not None:
                self._main_plot.plot(x, bg_y, pen=PEN_BACKGROUND)

            # Peaks (background + peak contribution)
            for idx, peak in enumerate(region.peaks):
                color = PEAK_COLORS[idx % len(PEAK_COLORS)]
                pen = pg.mkPen(color=color, width=2.5)
                self._main_plot.plot(x, bg_y + peak.y, pen=pen)
                # bg_y = bg_y + peak.y

            # Model
            self._main_plot.plot(x, region.model, pen=PEN_MODEL)

            # Collect residuals for bottom plot
            res = region.residuals
            if res.size > 0:
                all_res_x.append(x)
                all_res_y.append(res)

        if self._res_plot.isVisible():
            # Residuals subplot: same x as spectrum, one curve per region
            for rx, ry in zip(all_res_x, all_res_y):
                self._res_plot.plot(rx, ry, pen=PEN_RESIDUALS)

            # Lock residuals y-axis from data
            if all_res_y:
                concat = np.concatenate(all_res_y)
                r_min, r_max = float(np.min(concat)), float(np.max(concat))
                margin = max((r_max - r_min) * 0.1, 1e-12)
                self._res_plot.setYRange(r_min - margin, r_max + margin)
            else:
                self._res_plot.setYRange(-1, 1)
