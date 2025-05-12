import io
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime

from PySide6 import QtGui
from PySide6.QtWidgets import *
from PySide6.QtGui import QAction, QActionGroup

from app.sidebars import Sidebars
from app.canvas import PlotCanvas
from tools import Workspace

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_logging()

        self.setWindowTitle("XPS AI")
        self.setGeometry(100, 100, 1500, 750)

        m = self.load_model()
        self.workspace = Workspace(model=m)  # Initialize workspace

        self.initUI()
    
    def setup_logging(self, log_level=logging.DEBUG):
        self.logger = logging.getLogger(__name__)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler(sys.stdout)
        self.log_buffer = io.StringIO()
        memory_handler = logging.StreamHandler(self.log_buffer)

        console_handler.setFormatter(formatter)
        memory_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(memory_handler)

        self.logger.setLevel(log_level)
        self.logger.debug("Logging initialized.")

        sys.excepthook = self.handle_unhandled_exception

    def handle_unhandled_exception(self, exctype, value, tb):
        """Log unhandled exceptions with full traceback."""
        error_message = "".join(traceback.format_exception(exctype, value, tb))
        self.logger.critical(f"Unhandled Exception:\n{error_message}")

        now = datetime.now()
        file_path = f'error_log_{now.strftime("%Y%m%d_%H%M%S")}.txt'
        self.log_buffer.seek(0)  # Move to the start of the buffer
        log_contents = self.log_buffer.read()
        with open(file_path, "w") as f:
            f.write(log_contents)

    def load_model(self):
        self.logger.debug("Loading model")
        try:
            path = f'{sys._MEIPASS}/model.onnx' # For PyInstaller
        except Exception:
            main_dir = Path(sys.modules["__main__"].__file__).parent # For development and Nuitka
            path = main_dir.joinpath('model.onnx')
        return path

    def initUI(self):
        self.logger.debug("Initializing UI")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Toolbar for actions
        self.toolbar = Toolbar(self)
        self.addToolBar(self.toolbar)

        # Splitter for sidebar, main content, and right panel
        splitter = QSplitter()

        self.canvas = PlotCanvas(self, self.workspace)
        self.canvas.resize(800, 600) 
        self.sidebars = Sidebars(self, self.workspace, self.logger)

        splitter.addWidget(self.sidebars.left_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.sidebars.right_panel)
        self.sidebars.right_panel.layout().addWidget(self.canvas.cursor_label)

        # Set the splitter to the central widget
        central_layout = QHBoxLayout()
        central_layout.addWidget(splitter)
        central_widget.setLayout(central_layout)

        self.toolbar.setFocus()
    
    def update_viewer(self):
        spectrum = self.sidebars.current_spectrum
        region = self.sidebars.current_region
        self.logger.debug(f"Updating viewer with Spectrum: {spectrum}, Region: {region}")
        if spectrum is not None:
            self.canvas.reload_spectrum(spectrum)

            add_smoothing = self.toolbar.toggle_smoothed_data_action.isChecked()

            if spectrum is not None:
                if spectrum.is_predicted and self.toolbar.toggle_labeled_data_action.isChecked():
                    self.canvas.update_plot(disp_type='labeled')
                elif self.toolbar.toggle_raw_data_action.isChecked():
                    self.canvas.update_plot(disp_type='raw', smoothed=add_smoothing)
                else:
                    self.canvas.update_plot(disp_type='lines', smoothed=add_smoothing)

                    if region is not None:
                        self.logger.debug("Setting cursors")
                        self.canvas.load_cursors(region)

    def update_cursors(self, region):
        self.canvas.load_cursors(region)

    def update_sidebars(self):
        self.logger.debug("Updating sidebars")
        self.sidebars.update_spectra_tree()
    
    def load_spectra(self):
        self.logger.debug("Loading spectra")
        files, _ = QFileDialog.getOpenFileNames(self, "Load spectra", ".", "Data files (*.txt *.csv *.dat *.vms);;SPECS files (*.xml);;All files (*)")
        if files:
            self.workspace.load_files(*files)
            self.update_sidebars()
    
    def save_workspace(self):
        self.logger.debug("Saving workspace")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save workspace", ".", "Workspace files (*.pkl)")
        if file_path:
            self.workspace.save_workspace(file_path)
    
    def load_workspace(self):
        self.logger.debug("Loading workspace")
        file_path, _ = QFileDialog.getOpenFileName(self, "Load workspace", ".", "Workspace files (*.pkl)")
        if file_path:
            self.workspace.load_workspace(file_path)
            self.update_sidebars()

    def save_spectra(self):
        self.logger.debug("Saving spectra")
        folder_rpath = QFileDialog.getExistingDirectory(self, 'Select Folder')
        skip_survey = self.sidebars.skip_survey_box.isChecked()
        files, groups, spectra = self.sidebars.aggregate_left_panel_items()
        spectra = self.workspace.aggregate_unique_spectra(spectra, files, groups, skip_survey)
        if spectra is None or len(spectra) == 0:
            spectra = [self.sidebars.current_spectrum]
        if folder_rpath and spectra is not None:
            self.workspace.save_spectra(folder_rpath, spectra)
    
    def export_parameters(self):
        self.logger.debug("Exporting parameters")
        skip_survey = self.sidebars.skip_survey_box.isChecked()
        files, groups, spectra = self.sidebars.aggregate_left_panel_items()
        spectra = self.workspace.aggregate_unique_spectra(spectra, files, groups, skip_survey)
        if spectra is None or len(spectra) == 0:
            spectra = [self.sidebars.current_spectrum]
    
        if self.toolbar.toggle_aggregate_before_export.isChecked():
            file_path, _ = QFileDialog.getSaveFileName(self, "Save ", ".", "CSV Files (*.csv)")
            if file_path and spectra is not None:
                self.workspace.aggregate_and_export(file_path, spectra)
        else:
            folder_rpath = QFileDialog.getExistingDirectory(self, 'Select Folder')
            if folder_rpath and spectra is not None:
                self.workspace.export_params(folder_rpath, spectra)
    
    def save_logs(self):
        self.logger.debug("Saving logs")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Log File", "", "Text Files (*.txt);;All Files (*)")

        if file_path:  # If user selects a file
            self.log_buffer.seek(0)  # Move to the start of the buffer
            log_contents = self.log_buffer.read()

            with open(file_path, "w") as f:
                f.write(log_contents)

            self.logger.debug(f"All logs have been saved to {file_path}")

class Toolbar(QToolBar):
    def __init__(self, parent):
        super().__init__("Main Toolbar", parent)

        # Files Menu
        files_menu = QMenu("Files", parent)
        files_menu_action = self.addAction("Files")
        files_menu_action.setMenu(files_menu)
        files_menu_action.triggered.connect(lambda: files_menu.exec(self.mapToGlobal(self.rect().bottomLeft())))

        save_menu = QMenu("Save")
        files_menu.addMenu(save_menu)

        load_menu = QMenu("Load")
        files_menu.addMenu(load_menu)

        export_menu = QMenu("Export")
        files_menu.addMenu(export_menu)

        save_workspace_action = QAction("Save workspace", parent)
        save_workspace_action.triggered.connect(parent.save_workspace)
        save_menu.addAction(save_workspace_action)

        load_workspace_action = QAction("Load workspace", parent)
        load_workspace_action.triggered.connect(parent.load_workspace)
        load_menu.addAction(load_workspace_action)

        load_spectra_action = QAction("Load spectra", parent)
        load_spectra_action.triggered.connect(parent.load_spectra)
        load_menu.addAction(load_spectra_action)

        save_spectra_action = QAction('Export spectra', parent)
        save_spectra_action.triggered.connect(parent.save_spectra)
        export_menu.addAction(save_spectra_action)

        export_parameters_action = QAction('Export parameters', parent)
        export_parameters_action.triggered.connect(parent.export_parameters)
        export_menu.addAction(export_parameters_action)

        self.toggle_aggregate_before_export = QAction("Aggregate parameters into single file", parent, checkable=True)
        self.toggle_aggregate_before_export.setChecked(True)
        export_menu.addAction(self.toggle_aggregate_before_export)

        # View Menu
        view_menu = QMenu("View", parent)
        view_menu_action = self.addAction("View")
        view_menu_action.setMenu(view_menu)
        view_menu_action.triggered.connect(lambda: view_menu.exec(self.mapToGlobal(self.rect().bottomLeft())))
        
        # Spectra Viewer Options Group
        self.data_togglers_group = QActionGroup(parent)
        self.data_togglers_group.setExclusive(True)

        self.toggle_labeled_data_action = QAction("Labeled data", parent, checkable=True)
        self.toggle_labeled_data_action.triggered.connect(parent.update_viewer)
        view_menu.addAction(self.toggle_labeled_data_action)
        self.data_togglers_group.addAction(self.toggle_labeled_data_action)

        self.toggle_raw_data_action = QAction("Raw data", parent, checkable=True)
        self.toggle_raw_data_action.triggered.connect(parent.update_viewer)
        view_menu.addAction(self.toggle_raw_data_action)
        self.data_togglers_group.addAction(self.toggle_raw_data_action)

        self.toggle_lines_action = QAction("Peaks", parent, checkable=True)
        self.toggle_lines_action.triggered.connect(parent.update_viewer)
        view_menu.addAction(self.toggle_lines_action)
        self.data_togglers_group.addAction(self.toggle_lines_action)
        self.toggle_lines_action.setChecked(True) # On by default

        self.toggle_smoothed_data_action = QAction("Smoothed data", parent, checkable=True)
        self.toggle_smoothed_data_action.triggered.connect(parent.update_viewer)
        view_menu.addAction(self.toggle_smoothed_data_action)

        save_logs_action = QAction("Print logs", parent)
        save_logs_action.triggered.connect(parent.save_logs)
        self.addAction(save_logs_action)
