"""Shared spectrum- and region-level context menus for the plot and properties panel."""

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMenu, QWidget

from ..controller import ControllerWrapper
from ..dialogs.export_options_dialog import export_peaks, export_spectra
from ..dialogs.optimize_confirm import confirm_and_optimize

# TODO: refactor as a modular context menu factory


@dataclass
class SpectrumContextMenuActions:
    """
    Spectrum-level menu actions and enable/disable logic.

    Parameters
    ----------
    add_region : QAction
        Create a new region (default span) on the current spectrum.
    optimize : QAction
        Optimize all regions of the current spectrum.
    auto_fit : QAction
        Run the NN segmenter then optimize all regions of the current spectrum.
    """

    add_region: QAction
    optimize: QAction
    auto_fit: QAction
    export_spectrum_csv: QAction
    _controller: ControllerWrapper
    _dialog_parent: QWidget | None

    def update_enabled_state(self) -> None:
        """Enable or disable actions from current selection and region count."""
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            self.add_region.setEnabled(False)
            self.optimize.setEnabled(False)
            self.auto_fit.setEnabled(False)
            return

        has_regions = bool(self._controller.query.get_regions_ids(spectrum_id))

        self.add_region.setEnabled(True)
        self.optimize.setEnabled(has_regions)
        self.auto_fit.setEnabled(True)

    def _on_add_region(self) -> None:
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return
        self._controller.create_region(spectrum_id)

    def _on_optimize(self) -> None:
        optimize_from_selection(self._controller, self._dialog_parent)

    def _on_auto_fit(self) -> None:
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return
        self._controller.run_segmenter([spectrum_id])
        confirm_and_optimize(
            self._dialog_parent,
            self._controller,
            spectrum_ids=[spectrum_id],
        )

    def _on_export_spectrum_csv(self) -> None:
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return
        export_spectra(self._controller, [spectrum_id], parent=self._dialog_parent)


def optimize_from_selection(controller: ControllerWrapper, dialog_parent: QWidget | None) -> None:
    """
    Optimize the open spectrum, the selected region, or do nothing.

    All regions of the spectrum are optimized when a spectrum is open, no
    region is selected, and the spectrum has regions. Otherwise the selected
    region is optimized. With neither, this is a no-op.
    """
    spectrum_id = controller.selected_spectrum_id
    region_id = controller.selected_region_id
    if spectrum_id is not None and region_id is None:
        if controller.query.get_regions_ids(spectrum_id):
            confirm_and_optimize(
                dialog_parent,
                controller,
                spectrum_ids=[spectrum_id],
            )
        return
    if region_id is not None:
        confirm_and_optimize(
            dialog_parent,
            controller,
            region_ids=[region_id],
        )


def _action_with_shortcut(menu: QMenu, text: str, key: str) -> QAction:
    """Menu action that shows ``key`` without taking the window shortcut."""
    action = menu.addAction(text)
    action.setShortcut(QKeySequence(key))
    action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
    return action


def attach_spectrum_context_actions(
    menu: QMenu,
    controller: ControllerWrapper,
    dialog_parent: QWidget | None = None,
    *,
    include_model_actions: bool = True,
    on_enter_split_mode: Callable[[], None] | None = None,
    on_enter_add_peak_mode: Callable[[], None] | None = None,
) -> SpectrumContextMenuActions:
    """
    Append spectrum-level actions to ``menu`` and return handles for updates.

    Parameters
    ----------
    menu : QMenu
        Menu to populate.
    controller : ControllerWrapper
        Application controller.
    dialog_parent : QWidget or None, optional
        Parent for modal dialogs.
    include_model_actions : bool, optional
        When False, model-mutating actions are created but not added to ``menu``.
    on_enter_split_mode : callable or None, optional
        Enter interactive split-region plot mode.
    on_enter_add_peak_mode : callable or None, optional
        Enter interactive add-peak plot mode.

    Returns
    -------
    SpectrumContextMenuActions
        Action references and ``update_enabled_state`` for the current spectrum.
    """
    plot_modes = on_enter_split_mode is not None or on_enter_add_peak_mode is not None
    add_peak_at_point: QAction | None = None
    split_region: QAction | None = None
    if include_model_actions and plot_modes:
        if on_enter_add_peak_mode is not None:
            add_peak_at_point = _action_with_shortcut(menu, "Add peak at point", "A")
        optimize = _action_with_shortcut(menu, "Optimize", "O")
        if on_enter_split_mode is not None:
            split_region = _action_with_shortcut(menu, "Split region", "S")
        add_region = menu.addAction("Add region")
        auto_fit = menu.addAction("Auto fit")
    elif include_model_actions:
        add_region = menu.addAction("Add region")
        optimize = menu.addAction("Optimize")
        auto_fit = menu.addAction("Auto fit")
    else:
        add_region = QAction("Add region", menu)
        optimize = QAction("Optimize", menu)
        auto_fit = QAction("Auto fit", menu)
    export_spectrum_csv = menu.addAction("Export spectrum")

    state = SpectrumContextMenuActions(
        add_region=add_region,
        optimize=optimize,
        auto_fit=auto_fit,
        export_spectrum_csv=export_spectrum_csv,
        _controller=controller,
        _dialog_parent=dialog_parent,
    )
    if include_model_actions:
        add_region.triggered.connect(lambda _checked=False: state._on_add_region())
        optimize.triggered.connect(lambda _checked=False: state._on_optimize())
        auto_fit.triggered.connect(lambda _checked=False: state._on_auto_fit())
        if add_peak_at_point is not None and on_enter_add_peak_mode is not None:
            add_peak_at_point.triggered.connect(lambda _checked=False: on_enter_add_peak_mode())
        if split_region is not None and on_enter_split_mode is not None:
            split_region.triggered.connect(lambda _checked=False: on_enter_split_mode())
    export_spectrum_csv.triggered.connect(lambda _checked=False: state._on_export_spectrum_csv())

    return state


@dataclass
class RegionContextMenuActions:
    """
    Region-level menu actions and enable/disable logic.

    Parameters
    ----------
    add_peak : QAction
        Add a peak with the default model.
    set_background : QAction
        Add default background (disabled if one exists).
    optimize_region : QAction
        Optimize this region only.
    delete_region : QAction
        Remove the region.
    split_region : QAction or None
        Enter interactive split-region mode when wired.
    add_peak_at_point : QAction or None
        Enter interactive add-peak mode when wired.
    """

    add_peak: QAction
    set_background: QAction
    optimize_region: QAction
    delete_region: QAction
    export_peak_csv: QAction
    split_region: QAction | None
    add_peak_at_point: QAction | None
    _controller: ControllerWrapper
    _region_id: str
    _dialog_parent: QWidget
    _on_enter_split_mode: Callable[[], None] | None = None
    _on_enter_add_peak_mode: Callable[[], None] | None = None

    def update_enabled_state(self) -> None:
        """Disable set background when the region already has a background."""
        has_background = self._controller.query.get_background_id(self._region_id) is not None
        self.set_background.setEnabled(not has_background)

    def _on_add_peak(self) -> None:
        model_name = self._controller.get_app_parameters().default_peak_model
        self._controller.create_peak(self._region_id, model_name, parameters=None)

    def _on_set_background(self) -> None:
        model_name = self._controller.get_app_parameters().default_background_model
        self._controller.create_background(self._region_id, model_name, parameters=None)

    def _on_optimize_region(self) -> None:
        confirm_and_optimize(
            self._dialog_parent,
            self._controller,
            region_ids=[self._region_id],
        )

    def _on_delete_region(self) -> None:
        if self._controller.selected_region_id == self._region_id:
            self._controller.set_selection(self._controller.selected_spectrum_id, None)
        self._controller.full_remove_object(self._region_id)

    def _on_export_peak_csv(self) -> None:
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return
        export_peaks(self._controller, [spectrum_id], parent=self._dialog_parent)

    def _on_split_region(self) -> None:
        if self._on_enter_split_mode is not None:
            self._on_enter_split_mode()

    def _on_add_peak_at_point(self) -> None:
        if self._on_enter_add_peak_mode is not None:
            self._on_enter_add_peak_mode()


def attach_region_context_actions(
    menu: QMenu,
    controller: ControllerWrapper,
    region_id: str,
    dialog_parent: QWidget,
    *,
    include_model_actions: bool = True,
    on_enter_split_mode: Callable[[], None] | None = None,
    on_enter_add_peak_mode: Callable[[], None] | None = None,
) -> RegionContextMenuActions:
    """
    Append region-level actions to ``menu`` and return handles for updates.

    Parameters
    ----------
    menu : QMenu
        Menu to populate.
    controller : ControllerWrapper
        Application controller.
    region_id : str
        Target region identifier.
    dialog_parent : QWidget
        Parent for modal dialogs (e.g. optimize confirm).
    include_model_actions : bool, optional
        When False, model-mutating actions are created but not added to ``menu``.
    on_enter_split_mode : callable or None, optional
        Enter interactive split-region plot mode.
    on_enter_add_peak_mode : callable or None, optional
        Enter interactive add-peak plot mode.

    Returns
    -------
    RegionContextMenuActions
        Action references and ``update_enabled_state`` for this region.
    """
    plot_modes = on_enter_split_mode is not None or on_enter_add_peak_mode is not None
    split_region: QAction | None = None
    add_peak_at_point: QAction | None = None
    if include_model_actions and plot_modes:
        if on_enter_add_peak_mode is not None:
            add_peak_at_point = _action_with_shortcut(menu, "Add peak at point", "A")
        optimize_region = _action_with_shortcut(menu, "Optimize", "O")
        if on_enter_split_mode is not None:
            split_region = _action_with_shortcut(menu, "Split region", "S")
        add_peak = menu.addAction("Add peak")
        set_background = menu.addAction("Set background")
        delete_region = menu.addAction("Delete region")
    elif include_model_actions:
        add_peak = menu.addAction("Add peak")
        set_background = menu.addAction("Set background")
        optimize_region = menu.addAction("Optimize region")
        delete_region = menu.addAction("Delete region")
    else:
        add_peak = QAction("Add peak", menu)
        set_background = QAction("Set background", menu)
        optimize_region = QAction("Optimize region", menu)
        delete_region = QAction("Delete region", menu)
    export_peak_csv = menu.addAction("Export peaks")

    state = RegionContextMenuActions(
        add_peak=add_peak,
        set_background=set_background,
        optimize_region=optimize_region,
        delete_region=delete_region,
        export_peak_csv=export_peak_csv,
        split_region=split_region,
        add_peak_at_point=add_peak_at_point,
        _controller=controller,
        _region_id=region_id,
        _dialog_parent=dialog_parent,
        _on_enter_split_mode=on_enter_split_mode,
        _on_enter_add_peak_mode=on_enter_add_peak_mode,
    )
    if include_model_actions:
        add_peak.triggered.connect(lambda _checked=False: state._on_add_peak())
        set_background.triggered.connect(lambda _checked=False: state._on_set_background())
        if plot_modes:
            optimize_region.triggered.connect(
                lambda _checked=False: optimize_from_selection(
                    state._controller, state._dialog_parent
                )
            )
        else:
            optimize_region.triggered.connect(lambda _checked=False: state._on_optimize_region())
        delete_region.triggered.connect(lambda _checked=False: state._on_delete_region())
        if split_region is not None:
            split_region.triggered.connect(lambda _checked=False: state._on_split_region())
        if add_peak_at_point is not None:
            add_peak_at_point.triggered.connect(
                lambda _checked=False: state._on_add_peak_at_point()
            )
    export_peak_csv.triggered.connect(lambda _checked=False: state._on_export_peak_csv())

    return state
