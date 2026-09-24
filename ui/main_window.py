"""Main window: spectrum tree, plot area, properties, and menus."""

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QWidget,
)

from .assets import APP_NAME, load_app_icon
from .context_menus import optimize_from_selection
from .controller import ControllerWrapper
from .export_options_dialog import export_peaks, export_spectra
from .options_dialog import OptionsDialog
from .plot_area import PlotAreaWidget
from .properties_panel import PropertiesPanel
from .spectrum_tree_panel import SpectrumTreePanel


class MainWindow(QMainWindow):
    """
    Main application window hosting spectrum tree, plot area, and properties.

    The window wires menu actions to the :class:`ControllerWrapper`
    and listens to its signals to keep UI state (undo/redo, status bar, and
    future child widgets) in sync with the underlying model.

    Parameters
    ----------
    controller : ControllerWrapper
        Controller wrapper instance providing orchestrator access and Qt
        signals for collection and selection changes.
    parent : QWidget or None, optional
        Optional parent widget.
    """

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if getattr(sys, "frozen", False):
            self.setWindowIcon(load_app_icon())
        self._controller = controller

        self._action_new: QAction | None = None
        self._action_open: QAction | None = None
        self._action_save: QAction | None = None
        self._action_save_as: QAction | None = None
        self._action_exit: QAction | None = None
        self._action_export_spectrum_csv: QAction | None = None
        self._action_export_peak_csv: QAction | None = None
        self._action_export_all_selected_spectra_csv: QAction | None = None
        self._action_export_peaks_all_selected_spectra_csv: QAction | None = None
        self._action_undo: QAction | None = None
        self._action_redo: QAction | None = None
        self._action_optimize: QAction | None = None
        self._action_split_region: QAction | None = None
        self._action_add_peak_at_point: QAction | None = None
        self._action_load_nn_model: QAction | None = None
        self._action_app_parameters: QAction | None = None

        self._status_bar: QStatusBar | None = None

        self._spectrum_tree_panel: SpectrumTreePanel | None = None
        self._plot_area: PlotAreaWidget | None = None
        self._properties_panel: PropertiesPanel | None = None

        self._create_actions()
        self._create_menus()
        self._create_central_splitter()
        self._create_status_bar()
        self._connect_controller_signals()

        self._update_undo_redo_state(
            can_undo=self._controller.can_undo,
            can_redo=self._controller.can_redo,
        )
        self._update_window_title()
        self._update_status_bar()

        self.resize(1400, 720)

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------

    def _create_actions(self) -> None:
        """Create menu and toolbar actions."""
        self._action_new = QAction("New", self)
        self._action_open = QAction("Open…", self)
        self._action_open.setShortcut(QKeySequence.StandardKey.Open)
        self._action_save = QAction("Save", self)
        self._action_save.setShortcut(QKeySequence.StandardKey.Save)
        self._action_save_as = QAction("Save As…", self)
        self._action_exit = QAction("Exit", self)
        self._action_export_spectrum_csv = QAction("Export selected spectrum CSV…", self)
        self._action_export_peak_csv = QAction("Export selected region peak CSV…", self)
        self._action_export_all_selected_spectra_csv = QAction("Export all selected spectra…", self)
        self._action_export_peaks_all_selected_spectra_csv = QAction(
            "Export peaks from all selected spectra…", self
        )

        self._action_undo = QAction("Undo", self)
        self._action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self._action_redo = QAction("Redo", self)
        self._action_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self._action_undo.setEnabled(False)
        self._action_redo.setEnabled(False)

        self._action_split_region = QAction("Split region…", self)
        self._action_split_region.setCheckable(True)
        self._action_split_region.setShortcut(QKeySequence("S"))
        self._action_add_peak_at_point = QAction("Add peak at point…", self)
        self._action_add_peak_at_point.setCheckable(True)
        self._action_add_peak_at_point.setShortcut(QKeySequence("A"))
        self._action_optimize = QAction("Optimize", self)
        self._action_optimize.setShortcut(QKeySequence("O"))

        self._action_load_nn_model = QAction("Load NN model…", self)
        self._action_app_parameters = QAction("Application parameters…", self)

        self._action_new.triggered.connect(self._on_new_triggered)
        self._action_open.triggered.connect(self._on_open_triggered)
        self._action_save.triggered.connect(self._on_save_triggered)
        self._action_save_as.triggered.connect(self._on_save_as_triggered)
        self._action_exit.triggered.connect(self.close)
        self._action_export_spectrum_csv.triggered.connect(self._on_export_spectrum_csv_triggered)
        self._action_export_peak_csv.triggered.connect(self._on_export_peak_csv_triggered)
        self._action_export_all_selected_spectra_csv.triggered.connect(
            self._on_export_all_selected_spectra_triggered
        )
        self._action_export_peaks_all_selected_spectra_csv.triggered.connect(
            self._on_export_peaks_all_selected_spectra_triggered
        )

        self._action_undo.triggered.connect(self._on_undo_triggered)
        self._action_redo.triggered.connect(self._on_redo_triggered)
        self._action_split_region.triggered.connect(self._on_split_region_triggered)
        self._action_add_peak_at_point.triggered.connect(self._on_add_peak_at_point_triggered)
        self._action_optimize.triggered.connect(self._on_optimize_shortcut)

        self._action_load_nn_model.triggered.connect(self._on_load_nn_model_triggered)
        self._action_app_parameters.triggered.connect(self._on_app_parameters_triggered)

    def _create_menus(self) -> None:
        """Build the menu bar structure."""
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        if self._action_new is not None:
            file_menu.addAction(self._action_new)
        if self._action_open is not None:
            file_menu.addAction(self._action_open)
        if self._action_save is not None:
            file_menu.addAction(self._action_save)
        if self._action_save_as is not None:
            file_menu.addAction(self._action_save_as)
        export_menu = file_menu.addMenu("Export...")
        if self._action_export_spectrum_csv is not None:
            export_menu.addAction(self._action_export_spectrum_csv)
        if self._action_export_peak_csv is not None:
            export_menu.addAction(self._action_export_peak_csv)
        if self._action_export_all_selected_spectra_csv is not None:
            export_menu.addAction(self._action_export_all_selected_spectra_csv)
        if self._action_export_peaks_all_selected_spectra_csv is not None:
            export_menu.addAction(self._action_export_peaks_all_selected_spectra_csv)
        file_menu.addSeparator()
        if self._action_exit is not None:
            file_menu.addAction(self._action_exit)

        edit_menu = menu_bar.addMenu("Edit")
        if self._action_undo is not None:
            edit_menu.addAction(self._action_undo)
        if self._action_redo is not None:
            edit_menu.addAction(self._action_redo)

        if self._action_add_peak_at_point is not None:
            self.addAction(self._action_add_peak_at_point)
        if self._action_optimize is not None:
            self.addAction(self._action_optimize)
        if self._action_split_region is not None:
            self.addAction(self._action_split_region)

        options_menu = menu_bar.addMenu("Options")
        if self._action_load_nn_model is not None:
            options_menu.addAction(self._action_load_nn_model)
        if self._action_app_parameters is not None:
            options_menu.addAction(self._action_app_parameters)

    def _create_central_splitter(self) -> None:
        """Create the central splitter with left/center/right panels."""
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setObjectName("MainSplitter")
        splitter.setHandleWidth(1)

        self._spectrum_tree_panel = SpectrumTreePanel(self._controller, splitter)
        self._spectrum_tree_panel.setObjectName("SpectrumTreePanel")

        self._plot_area = PlotAreaWidget(self._controller, splitter)
        self._plot_area.setObjectName("PlotArea")

        self._properties_panel = PropertiesPanel(self._controller, splitter)
        self._properties_panel.setObjectName("PropertiesPanel")

        splitter.addWidget(self._spectrum_tree_panel)
        splitter.addWidget(self._plot_area)
        splitter.addWidget(self._properties_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([200, 430, 470])

        self.setCentralWidget(splitter)

        if self._plot_area is not None:
            self._plot_area.refresh()

    def _create_status_bar(self) -> None:
        """Create and attach the status bar."""
        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)
        self._status_bar = status_bar

    def _connect_controller_signals(self) -> None:
        """Connect controller wrapper signals to window slots."""
        self._controller.undoRedoStateChanged.connect(self._on_undo_redo_state_changed)
        self._controller.documentStateChanged.connect(self._on_document_state_changed)
        self._controller.selectionChanged.connect(self._on_selection_changed)

        if self._spectrum_tree_panel is not None:
            self._controller.spectrumHierarchyChanged.connect(self._spectrum_tree_panel.refresh)
            # Structure dots depend on regions/peaks created by auto-fit / optimize.
            self._controller.propertiesNeedsRefresh.connect(
                self._spectrum_tree_panel.tree.refresh_structure_status
            )
        if self._plot_area is not None:
            self._controller.plotNeedsRefresh.connect(self._plot_area.refresh)
            self._controller.selectionChanged.connect(self._plot_area.refresh)
            self._plot_area.editModeChanged.connect(self._on_plot_edit_mode_changed)
        if self._properties_panel is not None:
            self._controller.propertiesNeedsRefresh.connect(self._properties_panel.refresh)
            self._controller.selectionChanged.connect(
                self._properties_panel.on_controller_selection_changed
            )

    # ------------------------------------------------------------------
    # Slots for actions
    # ------------------------------------------------------------------

    def _on_new_triggered(self) -> None:
        """Create a new collection (clear current workspace)."""
        if not self._confirm_discard_changes():
            return
        self._controller.new_collection()

    def _on_open_triggered(self) -> None:
        """
        Open a collection or import one or more spectrum files.

        The dialog offers options to open a saved JSON collection or import
        spectra files supported by the import service (.txt, .csv, .dat, .vms,
        .vamas) via :meth:`ControllerWrapper.import_spectra`. Multiple spectrum
        files may be selected at once.
        """
        filenames, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Open or import",
            "",
            "Files (*.json *.txt *.csv *.dat *.vms *.vamas);;Collections (*.json);;"
            "Spectra (*.txt *.csv *.dat *.vms *.vamas);;All files (*)",
        )
        if not filenames:
            return

        spectrum_suffixes = {".txt", ".csv", ".dat", ".vms", ".vamas"}
        spectrum_paths = [
            name for name in filenames if Path(name).suffix.lower() in spectrum_suffixes
        ]
        collection_paths = [name for name in filenames if Path(name).suffix.lower() == ".json"]

        if spectrum_paths:
            self._controller.import_spectra(spectrum_paths)
        elif len(collection_paths) == 1:
            if not self._confirm_discard_changes():
                return
            self._controller.load_collection(collection_paths[0])
        else:
            self._show_info(
                "Nothing to open",
                "Select spectrum files to import, or a single JSON collection to open.",
            )
            return

        self._update_window_title()
        self._update_status_bar()

    def _on_save_triggered(self) -> None:
        """Save the collection using the default or last used path."""
        self._try_save()

    def _on_save_as_triggered(self) -> None:
        """Save the collection to a user-selected path."""
        self._try_save_as()

    def _on_undo_triggered(self) -> None:
        """Trigger an undo via the controller."""
        self._controller.undo()

    def _on_export_spectrum_csv_triggered(self) -> None:
        """Export currently selected spectrum as CSV."""
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            self._show_info("No spectrum selected", "Select a spectrum before exporting.")
            return
        export_spectra(self._controller, [spectrum_id], parent=self)

    def _on_export_peak_csv_triggered(self) -> None:
        """Export peak parameters from currently selected spectrum as CSV."""
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            self._show_info(
                "No spectrum selected", "Select a spectrum before exporting peak parameters."
            )
            return
        export_peaks(self._controller, [spectrum_id], parent=self)

    def _on_export_all_selected_spectra_triggered(self) -> None:
        """Export all selected spectra."""
        if self._spectrum_tree_panel is None:
            return
        spectrum_ids = self._spectrum_tree_panel.tree.get_selected_spectrum_ids()
        if not spectrum_ids:
            self._show_info("No spectrum selected", "Select one or more spectra before exporting.")
            return
        export_spectra(self._controller, spectrum_ids, parent=self)

    def _on_export_peaks_all_selected_spectra_triggered(self) -> None:
        """Export peak parameters from all selected spectra."""
        if self._spectrum_tree_panel is None:
            return
        spectrum_ids = self._spectrum_tree_panel.tree.get_selected_spectrum_ids()
        if not spectrum_ids:
            self._show_info(
                "No spectrum selected", "Select one or more spectra before exporting peaks."
            )
            return
        export_peaks(self._controller, spectrum_ids, parent=self)

    def _on_redo_triggered(self) -> None:
        """Trigger a redo via the controller."""
        self._controller.redo()

    def _on_optimize_shortcut(self) -> None:
        """Optimize the open spectrum, the selected region, or do nothing."""
        optimize_from_selection(self._controller, self)

    def _on_split_region_triggered(self, checked: bool = False) -> None:
        """Toggle interactive split-region mode on the plot."""
        if self._plot_area is None:
            return
        self._plot_area.set_edit_mode("split_region" if checked else None)

    def _on_add_peak_at_point_triggered(self, checked: bool = False) -> None:
        """Toggle interactive add-peak mode on the plot."""
        if self._plot_area is None:
            return
        self._plot_area.set_edit_mode("add_peak" if checked else None)

    def _on_plot_edit_mode_changed(self, mode: object) -> None:
        """Keep hidden shortcut actions in sync with the plot edit mode."""
        if self._action_split_region is not None:
            self._action_split_region.blockSignals(True)
            self._action_split_region.setChecked(mode == "split_region")
            self._action_split_region.blockSignals(False)
        if self._action_add_peak_at_point is not None:
            self._action_add_peak_at_point.blockSignals(True)
            self._action_add_peak_at_point.setChecked(mode == "add_peak")
            self._action_add_peak_at_point.blockSignals(False)

    def _on_load_nn_model_triggered(self) -> None:
        """Open a file dialog and load an NN model into the service."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load NN model",
            "",
            "ONNX models (*.onnx);;All files (*)",
        )
        if not filename:
            return

        self._controller.load_nn_model(filename)

    def _on_app_parameters_triggered(self) -> None:
        """Open the application parameters dialog."""
        params = self._controller.get_app_parameters()
        dialog = OptionsDialog(self)
        dialog.load_from_params(params)

        if not dialog.exec():
            return

        if not dialog.validate_and_apply(params):
            return

        self._controller.apply_app_parameters(params)

    # ------------------------------------------------------------------
    # Slots for controller signals
    # ------------------------------------------------------------------

    def _on_undo_redo_state_changed(self, can_undo: bool, can_redo: bool) -> None:
        """Update enabled state of undo/redo actions."""
        self._update_undo_redo_state(can_undo, can_redo)

    def _on_document_state_changed(self) -> None:
        """React to dirty flag or save path changes by refreshing title and status bar."""
        self._update_window_title()
        self._update_status_bar()

    def _on_selection_changed(
        self,
        spectrum_id: str | None,
        region_id: str | None,
        component_id: str | None = None,
    ) -> None:
        """
        React to selection changes by updating the status bar.

        Parameters
        ----------
        spectrum_id : str or None
            Selected spectrum identifier.
        region_id : str or None
            Selected region identifier.
        component_id : str or None, optional
            Selected component identifier.
        """
        del spectrum_id, region_id, component_id
        self._update_status_bar()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_undo_redo_state(self, can_undo: bool, can_redo: bool) -> None:
        """Enable or disable undo/redo actions based on controller state."""
        if self._action_undo is not None:
            self._action_undo.setEnabled(can_undo)
        if self._action_redo is not None:
            self._action_redo.setEnabled(can_redo)

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Prompt to save unsaved changes before the window closes.

        Parameters
        ----------
        event : QCloseEvent
            Qt close event; ignored if the user cancels or save fails.
        """
        if self._confirm_close():
            event.accept()
        else:
            event.ignore()

    def _try_save(self) -> bool:
        """
        Save the collection to the default path, or run Save As if unset.

        Returns
        -------
        bool
            True if the document was saved successfully.
        """
        if self._controller.get_default_save_path() is None:
            return self._try_save_as()
        self._controller.dump_collection()
        self._update_window_title()
        self._update_status_bar()
        return True

    def _try_save_as(self) -> bool:
        """
        Save the collection to a path chosen by the user.

        Returns
        -------
        bool
            True if the user picked a path and the document was saved.
        """
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save collection as",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not filename:
            return False

        self._controller.dump_collection(filename)
        self._update_window_title()
        self._update_status_bar()
        return True

    def _update_window_title(self) -> None:
        """Set the window title from save path and dirty state."""
        path: Path | None = self._controller.get_default_save_path()

        if path is None:
            name = "Untitled"
        else:
            name = path.name

        dirty = "*" if self._controller.is_dirty else ""
        self.setWindowTitle(f"{APP_NAME} - {dirty}{name}")

    def _update_status_bar(self) -> None:
        """Refresh the status bar text with path, dirty flag, and selection."""
        if self._status_bar is None:
            return

        path = self._controller.get_default_save_path()
        path_str = path.name if path is not None else "No file"

        spectrum_id = self._controller.selected_spectrum_id
        region_id = self._controller.selected_region_id
        component_id = self._controller.selected_component_id

        selection_parts: list[str] = []
        if spectrum_id is not None:
            selection_parts.append(f"Spectrum: {spectrum_id[:5]}")
        if region_id is not None:
            selection_parts.append(f"Region: {region_id[:5]}")
        if component_id is not None:
            selection_parts.append(f"Component: {component_id[:5]}")

        extra_selection = ""
        if self._spectrum_tree_panel is not None:
            selected_ids = self._spectrum_tree_panel.tree.get_selected_spectrum_ids()
            if selected_ids:
                if spectrum_id is not None and selected_ids.count(spectrum_id) > 0:
                    others = len(selected_ids) - 1
                else:
                    others = len(selected_ids)
                if others > 0:
                    extra_selection = f" (+{others} spectra)"

        selection_str_base = " | ".join(selection_parts) if selection_parts else "No selection"
        selection_str = f"{selection_str_base}{extra_selection}"

        if self._controller.is_dirty:
            text = f"* {path_str} | {selection_str}"
        else:
            text = f"{path_str} | {selection_str}"
        self._status_bar.showMessage(text)

    def _confirm_discard_changes(self) -> bool:
        """
        Ask the user to confirm discarding unsaved changes.

        Returns
        -------
        bool
            True if the operation may proceed, False to cancel.
        """
        if not self._controller.is_dirty:
            return True

        answer = QMessageBox.question(
            self,
            "Discard changes",
            "There may be unsaved changes. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_close(self) -> bool:
        """
        Ask whether to save, discard, or cancel when closing with dirty state.

        Returns
        -------
        bool
            True if the window may close, False to keep it open.
        """
        if not self._controller.is_dirty:
            return True

        choice = self._prompt_unsaved_close()
        if choice == "save":
            return self._try_save()
        if choice == "discard":
            return True
        return False

    def _prompt_unsaved_close(self) -> str:
        """
        Show the unsaved-changes close dialog.

        Returns
        -------
        {"save", "discard", "cancel"}
            User choice. Discard uses a short label so the button fits on Linux.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Unsaved changes")
        box.setText("Save changes before closing?")
        save_btn = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked == save_btn:
            return "save"
        if clicked == discard_btn:
            return "discard"
        return "cancel"

    def _show_info(self, title: str, message: str) -> None:
        """
        Display an informational message box.

        Parameters
        ----------
        title : str
            Dialog title.
        message : str
            Informative description.
        """
        QMessageBox.information(self, title, message)
