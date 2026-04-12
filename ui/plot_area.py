"""
pyqtgraph-based plot area for spectrum visualization.

Displays the selected spectrum with raw data, background, peaks, model,
and optional residuals. Driven by ``ControllerWrapper`` selection and signals,
using the viewer data provider protocol and :func:`tools.evaluation.spectrum_bundle`.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import HoverEvent, MouseClickEvent, MouseDragEvent
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QResizeEvent
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
PEAK_COLORS: list[str] = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


class DoubleClickAutoRangeViewBox(pg.ViewBox):
    """
    :class:`~pyqtgraph.ViewBox` in Rect mouse mode with double left-click auto-range.

    RectMode uses the left button to draw a zoom rectangle (single-button friendly).
    A double left-click resets the visible range via :meth:`~pyqtgraph.ViewBox.autoRange`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        **kwargs : Any
            Forwarded to :class:`~pyqtgraph.ViewBox`.
        """
        super().__init__(**kwargs)
        self.setMouseMode(pg.ViewBox.RectMode)

    def mouseClickEvent(self, ev: MouseClickEvent) -> None:
        """
        Reset range on double left-click; otherwise delegate to the base ViewBox.

        Parameters
        ----------
        ev : MouseClickEvent
            Scene click event from pyqtgraph.
        """
        if ev.button() == Qt.MouseButton.LeftButton and ev.double():
            ev.accept()
            self.autoRange()
            return
        super().mouseClickEvent(ev)


class VieBoxCustomContextMenu(DoubleClickAutoRangeViewBox):
    """
    ViewBox that shows spectrum-level context menu actions (export, etc.).

    Uses a lazily built :class:`~PySide6.QtWidgets.QMenu` tied to the application
    controller; menu visibility follows the ViewBox ``enableMenu`` state.
    """

    def __init__(
        self,
        *,
        controller: ControllerWrapper,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        controller : ControllerWrapper
            Application controller for spectrum context actions.
        **kwargs : Any
            Forwarded to :class:`DoubleClickAutoRangeViewBox`.
        """
        self._controller = controller
        self._spectrum_menu_actions: SpectrumContextMenuActions | None = None

        super().__init__(**kwargs)

    def _create_menu(self) -> QMenu:
        menu = QMenu()
        self._spectrum_menu_actions = attach_spectrum_context_actions(menu, self._controller, None)
        return menu

    def _applyMenuEnabled(self) -> None:
        enableMenu = self.state.get("enableMenu", True)

        if enableMenu and self.menu is None:
            self.menu = self._create_menu()

        elif not enableMenu and self.menu is not None:
            self.menu.setParent(None)
            self.menu = None
            self._spectrum_menu_actions = None

    def _update_menu_enabled_state(self) -> None:
        """Refresh enabled state of spectrum-level context actions."""
        if self._spectrum_menu_actions is not None:
            self._spectrum_menu_actions.update_enabled_state()

    def raiseContextMenu(self, ev: MouseClickEvent) -> None:
        """
        Show the custom spectrum menu at the event position.

        Parameters
        ----------
        ev : MouseClickEvent
            Click event that requested the context menu.
        """
        menu = self.getMenu(ev)
        if menu is not None:
            self._update_menu_enabled_state()
            menu.popup(ev.screenPos().toPoint())


class RegionContextPlotWidget(pg.PlotWidget):
    """
    Main spectrum :class:`~pyqtgraph.PlotWidget` with a custom ViewBox menu.

    Region-specific actions are handled on :class:`InteractiveRegion` items;
    the ViewBox menu exposes spectrum-level actions when the default
    pyqtgraph menu is enabled on the embedded :class:`VieBoxCustomContextMenu`.
    """

    def __init__(
        self,
        *,
        controller: ControllerWrapper,
        parent: QWidget | None = None,
    ) -> None:
        """
        Parameters
        ----------
        controller : ControllerWrapper
            Application controller passed to the custom ViewBox for menu actions.
        parent : QWidget or None, optional
            Parent widget.
        """
        self.menu: QMenu | None = None
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
    Vertical :class:`~pyqtgraph.LinearRegionItem` bound to a spectrum region id.

    Emits :attr:`sigClickedRegion` on left or right click (after handling
    double left-click for auto-range). Double left-click resets the host
    ViewBox with :meth:`~pyqtgraph.ViewBox.autoRange`.

    The region fill does not call ``acceptDrags`` for the left button (unlike
    the base class), so ViewBox pan and rectangle zoom work over the shaded
    band; edge :class:`~pyqtgraph.InfiniteLine` handles still claim drags.

    Attributes
    ----------
    sigClickedRegion : Signal
        Emits the region id (``str``) when the ROI receives a qualifying click.
    region_id : str
        Application model id for this ROI.
    menu : QMenu
        Context menu with region-level actions.
    """

    sigClickedRegion: Signal = Signal(str)

    def __init__(
        self,
        region_id: str,
        values: tuple[float, float],
        *,
        controller: ControllerWrapper,
        dialog_parent: QWidget,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        region_id : str
            Region id in the application model.
        values : tuple[float, float]
            Initial ``(min, max)`` slice in value mode coordinates.
        controller : ControllerWrapper
            Controller for context menu wiring.
        dialog_parent : QWidget
            Owner for dialogs raised from the menu.
        **kwargs : Any
            Passed through to :class:`~pyqtgraph.LinearRegionItem`.
        """
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
        )
        self._update_menu_enabled_state()

    def mouseClickEvent(self, ev: MouseClickEvent) -> None:
        """
        Select the region, show menus, or auto-range on double left-click.

        Parameters
        ----------
        ev : MouseClickEvent
            Click event from the graphics scene.
        """
        if ev.button() == Qt.MouseButton.LeftButton and ev.double():
            ev.accept()
            vb = self.getViewBox()
            if vb is not None:
                vb.autoRange()
            return
        if ev.button() == Qt.MouseButton.LeftButton or ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self.sigClickedRegion.emit(self.region_id)
            if ev.button() == Qt.MouseButton.RightButton:
                self.raiseContextMenu(ev)

    def hoverEvent(self, ev: HoverEvent) -> None:
        """
        Update hover highlighting without claiming left-button drags.

        The base :class:`~pyqtgraph.LinearRegionItem` registers left drags in
        hover, which forces the scene to deliver drags to this item and blocks
        ViewBox interaction. :class:`~pyqtgraph.InfiniteLine` children still
        accept drags for the two edges.

        Parameters
        ----------
        ev : HoverEvent
            pyqtgraph hover event for this item.
        """
        if self.movable and not ev.isExit():
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)

    def mouseDragEvent(self, ev: MouseDragEvent) -> None:
        """
        Decline body drags so the ViewBox can pan or zoom.

        Whole-region drag (moving the band without using the edge lines) is
        disabled intentionally.

        Parameters
        ----------
        ev : MouseDragEvent
            Drag event from the graphics scene.
        """
        ev.ignore()

    def _update_menu_enabled_state(self) -> None:
        """Refresh enabled state of region context actions."""
        self._region_menu_actions.update_enabled_state()

    def raiseContextMenu(self, ev: MouseClickEvent) -> bool:
        """
        Show the region context menu at the click position.

        Parameters
        ----------
        ev : MouseClickEvent
            Click event (typically right-button) requesting the menu.

        Returns
        -------
        bool
            Always ``True`` for callers that check for handled context menus.
        """
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

    Attributes
    ----------
    _main_plot : RegionContextPlotWidget
        Main spectrum and fit curves.
    _res_plot : pg.PlotWidget
        Residuals subplot, x-linked to the main ViewBox.
    """

    _main_plot: RegionContextPlotWidget
    _res_plot: pg.PlotWidget

    def __init__(
        self,
        controller: ControllerWrapper,
        parent: QWidget | None = None,
    ) -> None:
        """
        Parameters
        ----------
        controller : ControllerWrapper
            Source for selection, data, and parameters.
        parent : QWidget or None, optional
            Parent widget.
        """
        super().__init__(parent)
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
        self._res_plot = pg.PlotWidget(
            parent=self,
            viewBox=DoubleClickAutoRangeViewBox(enableMenu=False),
            enableMenu=False,
        )
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

    def _on_main_plot_mouse_moved(self, pos: QPointF) -> None:
        """
        Update the overlay label from the cursor position in scene coordinates.

        Parameters
        ----------
        pos : QPointF
            Cursor position in the graphics scene.
        """
        if self._cursor_label is None:
            return
        vb = self._main_plot.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            coord = vb.mapSceneToView(pos)
            self._cursor_label.setText(f"x: {coord.x():.4g}  y: {coord.y():.4g}")
        else:
            self._cursor_label.setText("x: —  y: —")

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Reposition the cursor overlay when the widget geometry changes.

        Parameters
        ----------
        event : QResizeEvent
            Qt resize event.
        """
        super().resizeEvent(event)
        self._position_cursor_label()

    def refresh(self) -> None:
        """
        Redraw plots from the controller's current spectrum selection.

        Loads data via ``get_spectrum_repr`` and :func:`spectrum_bundle`,
        updates ROIs, and repaints the main and residuals plots. If no spectrum
        is selected, clears the plot area.
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
        """Place the coordinate overlay in the upper-right of the main plot."""
        if self._cursor_label is not None:
            self._cursor_label.adjustSize()
            self._cursor_label.move(self._main_plot.width() - self._cursor_label.width() - 8, 8)

    def clear_plot(self) -> None:
        """Clear curve items, residuals, and region ROIs."""
        self._main_plot.clear()
        self._res_plot.clear()
        self._clear_rois()
        self._last_result = None
        self._last_spectrum_id = None

    def _clear_rois(self) -> None:
        """Detach every ``InteractiveRegion`` from the main plot."""
        for roi in self._roi_items_by_region.values():
            self._main_plot.removeItem(roi)
        self._roi_items_by_region.clear()
        self._roi_region_ids_in_plot.clear()

    def _sync_residuals_visibility(self) -> None:
        """Toggle residuals subplot visibility from application parameters."""
        params = self._controller.get_app_parameters()
        self._res_plot.setVisible(bool(getattr(params, "show_residuals_plot", True)))

    def _sync_rois_for_spectrum(self, *, spectrum_id: str) -> None:
        """
        Synchronize ``InteractiveRegion`` widgets with the model's regions.

        Creates or updates ROIs from region slices in ``"value"`` mode and
        removes items when regions are deleted.

        Parameters
        ----------
        spectrum_id : str
            Selected spectrum id whose regions should be reflected in the plot.
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
                self._main_plot.removeItem(roi)

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
        """
        Build a styled ``InteractiveRegion`` wired to controller signals.

        Parameters
        ----------
        region_id : str
            Model id for the region.
        start : float
            Lower slice bound in data units.
        stop : float
            Upper slice bound in data units.

        Returns
        -------
        InteractiveRegion
            ROI item not yet added to the plot (caller adds if needed).
        """
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
        """
        Update selection when the user clicks an ROI.

        Parameters
        ----------
        region_id : str
            Region id carried by the clicked ``InteractiveRegion``.
        """
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return
        self._controller.set_selection(spectrum_id, region_id)

    def _iter_rois(self) -> Iterable[InteractiveRegion]:
        """
        Yield every ``InteractiveRegion`` tracked for the current spectrum.

        Returns
        -------
        Iterable[InteractiveRegion]
            ROI items keyed by region id.
        """
        return self._roi_items_by_region.values()

    def _on_roi_region_change_finished(self, roi: InteractiveRegion) -> None:
        """
        Persist ROI bounds to the document after the user finishes dragging.

        Parameters
        ----------
        roi : InteractiveRegion
            Region item whose handles were moved.
        """
        low, high = roi.getRegion()
        self._controller.update_region_slice(roi.region_id, low, high, mode="value")

    def _draw_spectrum(self, result: SpectrumEvaluationResult) -> None:
        """
        Render the main spectrum stack and optional residuals subplot.

        Clears both plot widgets, re-attaches existing ``InteractiveRegion``
        items, draws the raw spectrum, per-region backgrounds, peaks, and model
        on the main plot, and draws per-region residual traces on ``_res_plot``
        when it is visible (y range derived from residual data).

        Parameters
        ----------
        result : SpectrumEvaluationResult
            Bundled x/y data and per-region fit components from
            :func:`spectrum_bundle`.
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
