"""
Shared spectrum- and region-level context menus for the plot and properties panel.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QWidget

from .component_creation_dialog import ComponentCreationDialog
from .controller import ControllerWrapper


@dataclass
class SpectrumContextMenuActions:
    """
    Spectrum-level menu actions and enable/disable logic.

    Parameters
    ----------
    add_region : QAction
        Create a new region (default span) on the current spectrum.
    optimize_regions : QAction
        Optimize all regions of the current spectrum.
    run_segmenter : QAction
        Run the NN segmenter on the current spectrum.
    """

    add_region: QAction
    optimize_regions: QAction
    run_segmenter: QAction
    _controller: ControllerWrapper

    def update_enabled_state(self) -> None:
        """Enable or disable actions from current selection and region count."""
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            self.add_region.setEnabled(False)
            self.optimize_regions.setEnabled(False)
            self.run_segmenter.setEnabled(False)
            return

        has_regions = bool(self._controller.query.get_regions_ids(spectrum_id))

        self.add_region.setEnabled(True)
        self.optimize_regions.setEnabled(has_regions)
        self.run_segmenter.setEnabled(not has_regions)

    def _on_add_region(self) -> None:
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return
        self._controller.create_region(spectrum_id)

    def _on_optimize_regions(self) -> None:
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return
        self._controller.optimize_regions(spectrum_ids=[spectrum_id])

    def _on_run_segmenter(self) -> None:
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return
        self._controller.run_segmenter([spectrum_id])


def attach_spectrum_context_actions(menu: QMenu, controller: ControllerWrapper) -> SpectrumContextMenuActions:
    """
    Append spectrum-level actions to ``menu`` and return handles for updates.

    Parameters
    ----------
    menu : QMenu
        Menu to populate.
    controller : ControllerWrapper
        Application controller.

    Returns
    -------
    SpectrumContextMenuActions
        Action references and ``update_enabled_state`` for the current spectrum.
    """
    add_region = menu.addAction("Add region")
    optimize_regions = menu.addAction("Optimize regions")
    run_segmenter = menu.addAction("Run segmenter")

    state = SpectrumContextMenuActions(
        add_region=add_region,
        optimize_regions=optimize_regions,
        run_segmenter=run_segmenter,
        _controller=controller,
    )
    add_region.triggered.connect(lambda _checked=False: state._on_add_region())
    optimize_regions.triggered.connect(lambda _checked=False: state._on_optimize_regions())
    run_segmenter.triggered.connect(lambda _checked=False: state._on_run_segmenter())

    return state


@dataclass
class RegionContextMenuActions:
    """
    Region-level menu actions and enable/disable logic.

    Parameters
    ----------
    add_peak : QAction
        Quick-add a default peak.
    set_background : QAction
        Quick-add default background (disabled if one exists).
    add_component : QAction
        Open the component creation dialog.
    optimize_region : QAction
        Optimize this region only.
    delete_region : QAction
        Remove the region.
    """

    add_peak: QAction
    set_background: QAction
    add_component: QAction
    optimize_region: QAction
    delete_region: QAction
    _controller: ControllerWrapper
    _region_id: str
    _dialog_parent: QWidget

    def update_enabled_state(self) -> None:
        """Disable set background when the region already has a background."""
        has_background = self._controller.query.get_background_id(self._region_id) is not None
        self.set_background.setEnabled(not has_background)

    def _on_add_peak(self) -> None:
        self._controller.create_peak(self._region_id, "pseudo-voigt", parameters=None)

    def _on_set_background(self) -> None:
        self._controller.create_background(self._region_id, "shirley", parameters=None)

    def _on_add_component(self) -> None:
        dialog = ComponentCreationDialog(
            self._controller,
            region_id=self._region_id,
            parent=self._dialog_parent,
        )
        dialog.exec()

    def _on_optimize_region(self) -> None:
        self._controller.optimize_regions(region_ids=[self._region_id])

    def _on_delete_region(self) -> None:
        if self._controller.selected_region_id == self._region_id:
            self._controller.set_selection(self._controller.selected_spectrum_id, None)
        self._controller.full_remove_object(self._region_id)


def attach_region_context_actions(
    menu: QMenu,
    controller: ControllerWrapper,
    region_id: str,
    dialog_parent: QWidget,
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
        Parent for modal dialogs (e.g. component creation).

    Returns
    -------
    RegionContextMenuActions
        Action references and ``update_enabled_state`` for this region.
    """
    add_peak = menu.addAction("Add peak (fast)")
    set_background = menu.addAction("Set background (fast)")
    add_component = menu.addAction("Add component...")
    optimize_region = menu.addAction("Optimize region")
    delete_region = menu.addAction("Delete region")

    state = RegionContextMenuActions(
        add_peak=add_peak,
        set_background=set_background,
        add_component=add_component,
        optimize_region=optimize_region,
        delete_region=delete_region,
        _controller=controller,
        _region_id=region_id,
        _dialog_parent=dialog_parent,
    )
    add_peak.triggered.connect(lambda _checked=False: state._on_add_peak())
    set_background.triggered.connect(lambda _checked=False: state._on_set_background())
    add_component.triggered.connect(lambda _checked=False: state._on_add_component())
    optimize_region.triggered.connect(lambda _checked=False: state._on_optimize_region())
    delete_region.triggered.connect(lambda _checked=False: state._on_delete_region())

    return state
