from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QWidget,
)

from app.error_dump import save_error_dump

from .controller import ControllerWrapper
from .options_dialog import OptionsDialog
from .plot_area import PlotAreaWidget
from .properties import PropertiesView
from .spectrum_tree_panel import SpectrumTreePanel


class MainWindow(QMainWindow):
    """
    Main application window hosting spectrum tree, plot area, and properties.

    The window wires menu and toolbar actions to the :class:`ControllerWrapper`
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
        self._controller = controller

        self._action_new: QAction | None = None
        self._action_open: QAction | None = None
        self._action_save: QAction | None = None
        self._action_save_as: QAction | None = None
        self._action_exit: QAction | None = None
        self._action_undo: QAction | None = None
        self._action_redo: QAction | None = None
        self._action_run_segmenter: QAction | None = None
        self._action_optimize_regions: QAction | None = None
        self._action_load_nn_model: QAction | None = None
        self._action_app_parameters: QAction | None = None

        self._main_toolbar: QToolBar | None = None
        self._status_bar: QStatusBar | None = None

        self._spectrum_tree_panel: SpectrumTreePanel | None = None
        self._plot_area: PlotAreaWidget | None = None
        self._properties_view: PropertiesView | None = None

        self._create_actions()
        self._create_menus()
        self._create_toolbar()
        self._create_central_splitter()
        self._create_status_bar()
        self._connect_controller_signals()

        self._update_undo_redo_state(
            can_undo=self._controller.orchestrator.can_undo,
            can_redo=self._controller.orchestrator.can_redo,
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
        self._action_save = QAction("Save", self)
        self._action_save_as = QAction("Save As…", self)
        self._action_exit = QAction("Exit", self)

        self._action_undo = QAction("Undo", self)
        self._action_redo = QAction("Redo", self)
        self._action_undo.setEnabled(False)
        self._action_redo.setEnabled(False)

        self._action_run_segmenter = QAction("Run segmenter", self)
        self._action_optimize_regions = QAction("Optimize regions", self)
        self._action_load_nn_model = QAction("Load NN model…", self)
        self._action_app_parameters = QAction("Application parameters…", self)

        self._action_new.triggered.connect(self._on_new_triggered)
        self._action_open.triggered.connect(self._on_open_triggered)
        self._action_save.triggered.connect(self._on_save_triggered)
        self._action_save_as.triggered.connect(self._on_save_as_triggered)
        self._action_exit.triggered.connect(self.close)

        self._action_undo.triggered.connect(self._on_undo_triggered)
        self._action_redo.triggered.connect(self._on_redo_triggered)

        self._action_run_segmenter.triggered.connect(self._on_run_segmenter_triggered)
        self._action_optimize_regions.triggered.connect(self._on_optimize_regions_triggered)
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
        file_menu.addSeparator()
        if self._action_exit is not None:
            file_menu.addAction(self._action_exit)

        edit_menu = menu_bar.addMenu("Edit")
        if self._action_undo is not None:
            edit_menu.addAction(self._action_undo)
        if self._action_redo is not None:
            edit_menu.addAction(self._action_redo)

        run_menu = menu_bar.addMenu("Run")
        if self._action_run_segmenter is not None:
            run_menu.addAction(self._action_run_segmenter)
        if self._action_optimize_regions is not None:
            run_menu.addAction(self._action_optimize_regions)

        options_menu = menu_bar.addMenu("Options")
        if self._action_load_nn_model is not None:
            options_menu.addAction(self._action_load_nn_model)
        if self._action_app_parameters is not None:
            options_menu.addAction(self._action_app_parameters)

    def _create_toolbar(self) -> None:
        """Create the main toolbar and add actions."""
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("MainToolbar")

        if self._action_open is not None:
            toolbar.addAction(self._action_open)
        if self._action_save is not None:
            toolbar.addAction(self._action_save)
        toolbar.addSeparator()
        if self._action_undo is not None:
            toolbar.addAction(self._action_undo)
        if self._action_redo is not None:
            toolbar.addAction(self._action_redo)

        toolbar.addSeparator()
        if self._action_run_segmenter is not None:
            toolbar.addAction(self._action_run_segmenter)
        if self._action_optimize_regions is not None:
            toolbar.addAction(self._action_optimize_regions)

        self.addToolBar(toolbar)
        self._main_toolbar = toolbar

    def _create_central_splitter(self) -> None:
        """Create the central splitter with left/center/right panels."""
        splitter = QSplitter(Qt.Horizontal, self)

        self._spectrum_tree_panel = SpectrumTreePanel(self._controller, splitter)
        self._spectrum_tree_panel.setObjectName("SpectrumTreePanel")

        self._plot_area = PlotAreaWidget(self._controller, splitter)
        self._plot_area.setObjectName("PlotArea")

        self._properties_view = PropertiesView(self._controller, splitter)
        self._properties_view.setObjectName("PropertiesView")

        splitter.addWidget(self._spectrum_tree_panel)
        splitter.addWidget(self._plot_area)
        splitter.addWidget(self._properties_view)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([200, 500, 400])

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
        if self._plot_area is not None:
            self._controller.plotNeedsRefresh.connect(self._plot_area.refresh)
            self._controller.selectionChanged.connect(self._plot_area.refresh)
        if self._properties_view is not None:
            self._controller.propertiesNeedsRefresh.connect(self._properties_view.refresh)
            self._controller.selectionChanged.connect(self._properties_view.refresh)

    # ------------------------------------------------------------------
    # Slots for actions
    # ------------------------------------------------------------------

    def _on_new_triggered(self) -> None:
        """Create a new collection (clear current workspace)."""
        if not self._confirm_discard_changes():
            return
        self._controller.orchestrator.new_collection()
        self._controller.emit_full_ui_refresh()

    def _on_open_triggered(self) -> None:
        """
        Open a collection or spectrum file using the controller.

        The dialog offers options to open a saved JSON collection or import
        spectra files supported by the import service (.txt, .dat, .vms,
        .vamas) via :meth:`ControllerWrapper.import_spectra`.
        """

        filename, selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open or import",
            "",
            "Files (*.json *.txt *.dat *.vms *.vamas);;Collections (*.json);;Spectra (*.txt *.dat *.vms *.vamas);;All files (*)",
        )
        if not filename:
            return

        try:
            suffix = Path(filename).suffix.lower()
            if "Spectra" in selected_filter or suffix in {".txt", ".dat", ".vms", ".vamas"}:
                self._controller.import_spectra(filename)
            else:
                if not self._confirm_discard_changes():
                    return
                self._controller.load_collection(filename)
        except Exception as exc:  # noqa: BLE001
            dump_path = save_error_dump(exc)
            message = f"{exc}\n\nDetails were saved to:\n{dump_path}"
            self._show_error("Failed to open file", message)
            return

        self._update_window_title()
        self._update_status_bar()

    def _on_save_triggered(self) -> None:
        """Save the collection using the default or last used path."""
        try:
            self._controller.dump_collection()
        except ValueError:
            self._on_save_as_triggered()
            return
        except Exception as exc:  # noqa: BLE001
            dump_path = save_error_dump(exc)
            message = f"{exc}\n\nDetails were saved to:\n{dump_path}"
            self._show_error("Failed to save file", message)
            return

        self._update_window_title()
        self._update_status_bar()

    def _on_save_as_triggered(self) -> None:
        """Save the collection to a user-selected path."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save collection as",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not filename:
            return

        try:
            self._controller.dump_collection(filename)
        except Exception as exc:  # noqa: BLE001
            dump_path = save_error_dump(exc)
            message = f"{exc}\n\nDetails were saved to:\n{dump_path}"
            self._show_error("Failed to save file", message)
            return

        self._update_window_title()
        self._update_status_bar()

    def _on_undo_triggered(self) -> None:
        """Trigger an undo via the controller."""
        try:
            self._controller.undo()
        except Exception as exc:  # noqa: BLE001
            dump_path = save_error_dump(exc)
            message = f"{exc}\n\nDetails were saved to:\n{dump_path}"
            self._show_error("Failed to undo", message)

    def _on_redo_triggered(self) -> None:
        """Trigger a redo via the controller."""
        try:
            self._controller.redo()
        except Exception as exc:  # noqa: BLE001
            dump_path = save_error_dump(exc)
            message = f"{exc}\n\nDetails were saved to:\n{dump_path}"
            self._show_error("Failed to redo", message)

    def _on_run_segmenter_triggered(self) -> None:
        """Run the segmenter for the currently selected spectrum."""
        spectrum_ids: list[str] = []
        if self._spectrum_tree_panel is not None:
            spectrum_ids = self._spectrum_tree_panel.tree.get_selected_spectrum_ids()
        if not spectrum_ids:
            self._show_info("No spectrum selected", "Select a spectrum before running the segmenter.")
            return

        try:
            self._controller.run_segmenter(spectrum_ids)
        except Exception as exc:  # noqa: BLE001
            dump_path = save_error_dump(exc)
            message = f"{exc}\n\nDetails were saved to:\n{dump_path}"
            self._show_error("Failed to run segmenter", message)

    def _on_optimize_regions_triggered(self) -> None:
        """Run optimization for regions associated with the selected spectrum."""
        spectrum_ids: list[str] = []
        if self._spectrum_tree_panel is not None:
            spectrum_ids = self._spectrum_tree_panel.tree.get_selected_spectrum_ids()
        if not spectrum_ids:
            self._show_info("No spectrum selected", "Select a spectrum before optimizing regions.")
            return

        try:
            self._controller.optimize_regions(spectrum_ids=spectrum_ids)
        except Exception as exc:  # noqa: BLE001
            dump_path = save_error_dump(exc)
            message = f"{exc}\n\nDetails were saved to:\n{dump_path}"
            self._show_error("Failed to optimize regions", message)

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

        try:
            self._controller.load_nn_model(filename)
        except Exception as exc:  # noqa: BLE001
            dump_path = save_error_dump(exc)
            message = f"{exc}\n\nDetails were saved to:\n{dump_path}"
            self._show_error("Failed to load NN model", message)

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

    def _on_selection_changed(self, spectrum_id: str | None, region_id: str | None) -> None:
        """
        React to selection changes by updating the status bar.

        Parameters
        ----------
        spectrum_id : str or None
            Selected spectrum identifier.
        region_id : str or None
            Selected region identifier.
        """
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

    def _update_window_title(self) -> None:
        """Set the window title based on save path"""
        path: Path | None = self._controller.get_default_save_path()

        if path is None:
            name = "Untitled"
        else:
            name = path.name

        title = f"Spectrum Viewer - {name}"
        self.setWindowTitle(title)

    def _update_status_bar(self) -> None:
        """Refresh the status bar text with path, dirty flag, and selection."""
        if self._status_bar is None:
            return

        path = self._controller.get_default_save_path()
        path_str = path.name if path is not None else "No file"

        spectrum_id = self._controller.selected_spectrum_id
        region_id = self._controller.selected_region_id

        selection_parts: list[str] = []
        if spectrum_id is not None:
            selection_parts.append(f"Spectrum: {spectrum_id[:5]}")
        if region_id is not None:
            selection_parts.append(f"Region: {region_id[:5]}")

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

        if self._controller.orchestrator.is_dirty:
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
        if not self._controller.orchestrator.is_dirty:
            return True

        answer = QMessageBox.question(
            self,
            "Discard changes",
            "There may be unsaved changes. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _show_error(self, title: str, message: str) -> None:
        """
        Display an error message box.

        Parameters
        ----------
        title : str
            Dialog title.
        message : str
            Error description.
        """
        QMessageBox.critical(self, title, message)

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
