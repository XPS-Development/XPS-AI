"""
pyqtgraph-based plot area for spectrum visualization.

Displays the selected spectrum with raw data, background, peaks, model,
and an optional chi-squared subplot. Driven by ``ControllerWrapper`` selection and signals,
using precomputed plot data from the application query layer.
"""

from collections.abc import Iterable
from typing import Any, Protocol, cast

import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import HoverEvent, MouseClickEvent, MouseDragEvent
from PySide6.QtCore import QPointF, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QLabel, QMenu, QVBoxLayout, QWidget

from core.evaluation import PlotCurve, SpectrumPlotData

from .component_colors import color_for_component
from .context_menus import (
    SpectrumContextMenuActions,
    attach_region_context_actions,
    attach_spectrum_context_actions,
)
from .controller import ControllerWrapper

# Curve styling constants
PEN_BACKGROUND = pg.mkPen(color="k", width=1, style=Qt.PenStyle.DashLine)
PEN_BACKGROUND_SELECTED = pg.mkPen(color="k", width=3, style=Qt.PenStyle.DashLine)
PEN_MODEL = pg.mkPen(color="r", width=1.5)
PEN_RESIDUALS = pg.mkPen(color="#808080", width=2)

REGION_BOUNDS_PEN = pg.mkPen(color="#000000", width=3)
REGION_BOUNDS_HOVER_PEN = pg.mkPen(color="#000000", width=4)
REGION_BOUNDS_BRUSH = pg.mkBrush(0, 0, 0, 0)
REGION_BOUNDS_HOVER_BRUSH = pg.mkBrush(0, 0, 255, 10)

_PEAK_WIDTH = 4.0
_PEAK_WIDTH_SELECTED = 6.0
_PEAK_WIDTH_DIMMED = 2.5
_CURVE_CLICK_WIDTH = 12
_RAW_SYMBOL_SIZE = 4
_RAW_COLOR = (140, 140, 140, 220)
# Fixed left-axis width so main and residuals plot areas stay aligned.
_LEFT_AXIS_WIDTH = 80


class DoubleClickAutoRangeViewBox(pg.ViewBox):
    """
    :class:`~pyqtgraph.ViewBox` in Rect mouse mode with double left-click auto-range.

    RectMode uses the left button to draw a zoom rectangle (single-button friendly).
    A double left-click resets the visible range via :meth:`~pyqtgraph.ViewBox.autoRange`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the view box.

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
        """Initialize the view box with a spectrum context menu.

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
        """Initialize the plot widget with a custom ViewBox.

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

        plot_item = self.plotItem
        assert plot_item is not None
        vb = plot_item.getViewBox()
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
        """Initialize the linear region item for a spectrum region.

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
    chi-squared subplot with shared x-axis and locked y-axis, optional ROI
    for the selected region, cursor (x, y) overlay, and context menu for
    region-aware actions. Refreshes on collection or selection changes via
    the connected controller signals.

    Attributes
    ----------
    _main_plot : RegionContextPlotWidget
        Main spectrum and fit curves.
    _res_plot : pg.PlotWidget
        Chi-squared subplot, x-linked to the main ViewBox.
    """

    _main_plot: RegionContextPlotWidget
    _res_plot: pg.PlotWidget

    def __init__(
        self,
        controller: ControllerWrapper,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the plot area from the controller.

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
        self._last_plot_data: SpectrumPlotData | None = None
        self._last_spectrum_id: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main plot (spectrum, background, peaks, model)
        self._main_plot = RegionContextPlotWidget(controller=self._controller)
        self._main_plot.setBackground("w")
        self._main_plot.showGrid(x=True, y=True, alpha=0.3)
        self._main_plot.getAxis("left").setWidth(_LEFT_AXIS_WIDTH)
        layout.addWidget(self._main_plot, stretch=1)

        # Chi-squared plot (shared x-axis, locked y)
        self._res_plot = pg.PlotWidget(
            parent=self,
            viewBox=DoubleClickAutoRangeViewBox(enableMenu=False),
            enableMenu=False,
        )
        self._res_plot.setBackground("w")
        self._res_plot.showGrid(x=True, y=True, alpha=0.3)
        self._res_plot.setMinimumHeight(80)
        self._res_plot.getAxis("left").setWidth(_LEFT_AXIS_WIDTH)
        self._res_plot.getViewBox().setXLink(self._main_plot.getViewBox())
        self._res_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        layout.addWidget(self._res_plot, stretch=0)

        self._chi_label = QLabel(self._res_plot)
        self._chi_label.setObjectName("PlotChiSquareLabel")
        self._chi_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._chi_label.hide()

        # Cursor (x, y) overlay on main plot
        self._cursor_label = QLabel(self._main_plot)
        self._cursor_label.setObjectName("PlotCursorLabel")
        self._cursor_label.setText("x: —  y: —")
        self._cursor_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        class _SceneWithMouseSignal(Protocol):
            sigMouseMoved: Any

        scene = cast(_SceneWithMouseSignal, self._main_plot.scene())
        scene.sigMouseMoved.connect(self._on_main_plot_mouse_moved)

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
        self._position_cursor_label()

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
        self._position_chi_label()

    def refresh(self) -> None:
        """
        Redraw plots from the controller's current spectrum selection.

        Loads data via :meth:`QueryService.get_spectrum_plot_data`,
        updates ROIs, and repaints the main and chi-squared plots. If no spectrum
        is selected, clears the plot area.
        """
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            self.clear_plot()
            return

        plot_data = self._controller.query.get_spectrum_plot_data(spectrum_id, normalized=False)

        self._last_plot_data = plot_data
        self._sync_residuals_visibility()
        self._sync_rois_for_spectrum(spectrum_id=spectrum_id)
        self._draw_spectrum(plot_data)
        self._update_chi_label(plot_data)
        self._position_cursor_label()

    def _viewbox_rect(self, plot: pg.PlotWidget) -> QRect:
        """
        Return the data area of ``plot`` in that widget's coordinates.

        Falls back to the area right of the fixed left axis when the view box
        has not been laid out yet.
        """
        scene_rect = plot.getViewBox().sceneBoundingRect()
        if scene_rect.width() >= 1.0 and scene_rect.height() >= 1.0:
            top_left = plot.mapFromScene(scene_rect.topLeft())
            bottom_right = plot.mapFromScene(scene_rect.bottomRight())
            rect = QRect(top_left, bottom_right).normalized()
            if rect.width() >= 1 and rect.height() >= 1:
                return rect
        return QRect(
            _LEFT_AXIS_WIDTH,
            0,
            max(plot.width() - _LEFT_AXIS_WIDTH, 0),
            max(plot.height(), 1),
        )

    def _position_cursor_label(self) -> None:
        """Keep the coordinate readout inside the main plot's data area, top-right."""
        if self._cursor_label is None:
            return
        self._cursor_label.adjustSize()
        rect = self._viewbox_rect(self._main_plot)
        x = rect.right() - self._cursor_label.width() - 6
        self._cursor_label.move(max(x, rect.left() + 4), rect.top() + 4)

    def _position_chi_label(self) -> None:
        """Keep the χ² readout inside the error plot's data area, clear of the axis."""
        if not self._chi_label.isVisible():
            return
        self._chi_label.adjustSize()
        rect = self._viewbox_rect(self._res_plot)
        x = rect.left() + 6
        max_x = rect.right() - self._chi_label.width() - 4
        if max_x >= rect.left():
            x = min(x, max_x)
        self._chi_label.move(x, rect.top() + 2)

    def _update_chi_label(self, plot_data: SpectrumPlotData) -> None:
        """Show the summed χ² criterion for the curves on the error plot."""
        stat = plot_data.chi_square
        if stat is None or not self._res_plot.isVisible():
            self._chi_label.hide()
            return
        reduced = stat.reduced_chi_square
        if reduced is None:
            text = f"χ² = {stat.chi_square:.6g}"
        else:
            text = f"χ² = {stat.chi_square:.6g}    χ²/dof = {reduced:.4g}"
        self._chi_label.setText(text)
        self._chi_label.show()
        self._position_chi_label()

    def clear_plot(self) -> None:
        """Clear curve items, the chi-squared subplot, and region ROIs."""
        self._main_plot.clear()
        self._res_plot.clear()
        self._chi_label.hide()
        self._clear_rois()
        self._last_plot_data = None
        self._last_spectrum_id = None

    def _clear_rois(self) -> None:
        """Detach every ``InteractiveRegion`` from the main plot."""
        for roi in self._roi_items_by_region.values():
            self._main_plot.removeItem(roi)
        self._roi_items_by_region.clear()
        self._roi_region_ids_in_plot.clear()

    def _sync_residuals_visibility(self) -> None:
        """Toggle the chi-squared subplot visibility from application parameters."""
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
                roi = self._create_region_roi(
                    region_id=rid, start=float(start_val), stop=float(stop_val)
                )
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
        roi.sigRegionChangeFinished.connect(
            lambda _roi=roi: self._on_roi_region_change_finished(_roi)
        )
        roi.sigClickedRegion.connect(self._on_roi_clicked)
        return roi

    def _on_roi_clicked(self, region_id: str) -> None:
        """
        Update selection when the user clicks an ROI.

        Deferred so a plot refresh does not destroy the ROI while its click
        handler is still running.

        Parameters
        ----------
        region_id : str
            Region id carried by the clicked ``InteractiveRegion``.
        """
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return

        def _apply() -> None:
            self._controller.set_selection(spectrum_id, region_id)

        QTimer.singleShot(0, _apply)

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

    def _draw_spectrum(self, plot_data: SpectrumPlotData) -> None:
        """
        Render the main spectrum stack and optional chi-squared subplot.

        Clears both plot widgets, re-attaches existing ``InteractiveRegion``
        items, and draws precomputed curves on the main and chi-squared plots.

        Parameters
        ----------
        plot_data : SpectrumPlotData
            Display-ready curves from the application query layer.
        """
        self._main_plot.clear()
        self._res_plot.clear()

        # Re-add ROIs after clearing to keep them visible, but only once per redraw.
        self._roi_region_ids_in_plot.clear()
        for rid, roi in self._roi_items_by_region.items():
            self._main_plot.addItem(roi)
            self._roi_region_ids_in_plot.add(rid)

        for curve in plot_data.curves:
            if curve.kind == "residual":
                if self._res_plot.isVisible():
                    self._res_plot.plot(curve.x, curve.y, pen=self._pen_for_curve(curve))
                continue

            if curve.kind == "raw":
                item = self._main_plot.plot(
                    curve.x,
                    curve.y,
                    pen=pg.mkPen(color=_RAW_COLOR, width=1),
                    symbol="o",
                    symbolSize=_RAW_SYMBOL_SIZE,
                    symbolPen=None,
                    symbolBrush=_RAW_COLOR,
                )
                item.setZValue(0)
                continue

            pen = self._pen_for_curve(curve)
            item = self._main_plot.plot(curve.x, curve.y, pen=pen)
            if curve.kind == "model":
                item.setZValue(30)
                continue
            if curve.kind == "background":
                selected = curve.component_id is not None and self._is_selected_component(
                    curve.component_id
                )
                item.setZValue(20 if selected else 5)
                if curve.component_id is not None:
                    item.setCurveClickable(True, width=_CURVE_CLICK_WIDTH)
                    cid = curve.component_id
                    item.sigClicked.connect(
                        lambda _item, _ev, component_id=cid: self._on_curve_clicked(component_id)
                    )
                continue
            if curve.kind == "peak" and curve.component_id is not None:
                item.setZValue(20 if self._is_selected_component(curve.component_id) else 10)
                item.setCurveClickable(True, width=_CURVE_CLICK_WIDTH)
                cid = curve.component_id
                item.sigClicked.connect(
                    lambda _item, _ev, component_id=cid: self._on_curve_clicked(component_id)
                )

        if self._res_plot.isVisible():
            if plot_data.residual_y_range is not None:
                self._res_plot.setYRange(*plot_data.residual_y_range)
            else:
                self._res_plot.setYRange(-1, 1)

        self._align_plot_axes()

    def _align_plot_axes(self) -> None:
        """Keep main and chi-squared left axes the same width so plot areas line up."""
        self._main_plot.getAxis("left").setWidth(_LEFT_AXIS_WIDTH)
        self._res_plot.getAxis("left").setWidth(_LEFT_AXIS_WIDTH)

    def _is_selected_component(self, component_id: str) -> bool:
        """Return True if ``component_id`` is the controller's selected component."""
        return self._controller.selected_component_id == component_id

    def _on_curve_clicked(self, component_id: str) -> None:
        """
        Select the component whose curve was clicked.

        Selection is deferred so :meth:`refresh` (triggered by
        ``selectionChanged``) does not destroy the clicked ``PlotDataItem``
        while ``sigClicked`` is still on the stack.

        Parameters
        ----------
        component_id : str
            Peak or background id carried by the clicked plot item.
        """
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return
        try:
            dto = self._controller.query.get_component_dto(component_id)
        except KeyError:
            return
        region_id = dto.parent_id

        def _apply() -> None:
            self._controller.set_selection(spectrum_id, region_id, component_id)

        QTimer.singleShot(0, _apply)

    def _pen_for_curve(self, curve: PlotCurve) -> Any:
        """Map a plot curve kind to a pyqtgraph pen, with selection highlighting."""
        selected_id = self._controller.selected_component_id
        is_selected = curve.component_id is not None and curve.component_id == selected_id
        has_selection = selected_id is not None

        if curve.kind == "background":
            return PEN_BACKGROUND_SELECTED if is_selected else PEN_BACKGROUND
        if curve.kind == "model":
            return PEN_MODEL
        if curve.kind == "peak":
            color = color_for_component(kind="peak", index=curve.peak_index or 0)
            if is_selected:
                width = _PEAK_WIDTH_SELECTED
            elif has_selection:
                width = _PEAK_WIDTH_DIMMED
            else:
                width = _PEAK_WIDTH
            return pg.mkPen(color=color, width=width)
        return PEN_RESIDUALS
