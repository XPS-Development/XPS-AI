import io
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime
from itertools import chain

import numpy as np
import pyqtgraph as pg

from PySide6 import QtGui
from PySide6.QtWidgets import *
from PySide6.QtGui import QAction, QPalette, QColor, QActionGroup
from PySide6.QtCore import Qt, QThread, Signal

from tools import Workspace


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_logging()

        self.setWindowTitle("Spectral Data Processing App")
        self.setGeometry(100, 100, 1500, 750)

        m = self.load_model()
        self.workspace = Workspace(model=m)  # Initialize your workspace

        self.initUI()
    
    def setup_logging(self, log_level=logging.DEBUG):
        # Set up logging
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
        color = self.palette().color(QtGui.QPalette.Base)
        self.canvas.setBackground(color)
        self.sidebars = Sidebars(self, self.workspace, self.logger)

        splitter.addWidget(self.sidebars.left_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.sidebars.right_panel)

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
        files, _ = QFileDialog.getOpenFileNames(self, "Load Spectra", ".", "All Files (*);;Vamas Files (*.vms);;Text Files (*.txt);;SPECS Files (*.xml)")
        if files:
            self.workspace.load_files(*files)
            self.update_sidebars()
    
    def save_workspace(self):
        self.logger.debug("Saving workspace")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Workspace", ".", "Workspace files (*.pkl)")
        if file_path:
            self.workspace.save_workspace(file_path)
    
    def load_workspace(self):
        self.logger.debug("Loading workspace")
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Workspace", ".", "Workspace files (*.pkl)")
        if file_path:
            self.workspace.load_workspace(file_path)
            self.update_sidebars()

    def save_spectra(self):
        self.logger.debug("Saving spectra")
        folder_rpath = QFileDialog.getExistingDirectory(self, 'Select Folder')
        spectra = self.sidebars.get_selected_spectra()
        if spectra is None or len(spectra) == 0:
            spectra = [self.sidebars.current_spectrum]
        if folder_rpath and spectra is not None:
            self.workspace.save_spectra(folder_rpath, spectra)
    
    def export_parameters(self):
        self.logger.debug("Exporting parameters")
        spectra = self.sidebars.get_selected_spectra()
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

    def change_prediction_threshold(self):
        self.logger.debug("Changing prediction threshold")
        value = self.workspace.pred_threshold
        new_threshold, ok = QInputDialog.getDouble(self, "Change Prediction Threshold", "Enter new prediction threshold:", value, 0, 1, 2, step=0.01)
        if ok:
            self.workspace.set_prediction_threshold(new_threshold)

        self.update_viewer()
    
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
        files_menu = QMenu("Files...", parent)
        files_menu_action = self.addAction("Files...")
        files_menu_action.setMenu(files_menu)
        files_menu_action.triggered.connect(lambda: files_menu.exec(self.mapToGlobal(self.rect().bottomLeft())))
        
        save_workspace_action = QAction("Save Workspace", parent)
        save_workspace_action.triggered.connect(parent.save_workspace)
        files_menu.addAction(save_workspace_action)

        load_workspace_action = QAction("Load Workspace", parent)
        load_workspace_action.triggered.connect(parent.load_workspace)
        files_menu.addAction(load_workspace_action)

        load_spectra_action = QAction("Load Spectra", parent)
        load_spectra_action.triggered.connect(parent.load_spectra)
        files_menu.addAction(load_spectra_action)

        save_spectra_action = QAction('Export Spectra', parent)
        save_spectra_action.triggered.connect(parent.save_spectra)
        files_menu.addAction(save_spectra_action)

        export_parameters_action = QAction('Export Parameters', parent)
        export_parameters_action.triggered.connect(parent.export_parameters)
        files_menu.addAction(export_parameters_action)

        # Options Menu
        options_menu = QMenu("Options...", parent)
        options_action = self.addAction("Options...")
        options_action.setMenu(options_menu)
        options_action.triggered.connect(lambda: options_menu.exec(self.mapToGlobal(self.rect().bottomLeft())))

        # Change Prediction Threshold Action
        change_threshold_action = QAction("Change prediction threshold", parent)
        change_threshold_action.triggered.connect(parent.change_prediction_threshold)
        options_menu.addAction(change_threshold_action)

        # Spectra Viewer Options Group
        self.data_togglers_group = QActionGroup(parent)
        self.data_togglers_group.setExclusive(True)

        # Toggle Labeled Data Action
        self.toggle_labeled_data_action = QAction("Show labeled data", parent, checkable=True)
        self.toggle_labeled_data_action.triggered.connect(parent.update_viewer)
        options_menu.addAction(self.toggle_labeled_data_action)
        self.data_togglers_group.addAction(self.toggle_labeled_data_action)

        # Toggle Raw Action
        self.toggle_raw_data_action = QAction("Show raw data", parent, checkable=True)
        self.toggle_raw_data_action.triggered.connect(parent.update_viewer)
        options_menu.addAction(self.toggle_raw_data_action)
        self.data_togglers_group.addAction(self.toggle_raw_data_action)

        # Toggle Line Data Action
        self.toggle_lines_action = QAction("Show lines", parent, checkable=True)
        self.toggle_lines_action.triggered.connect(parent.update_viewer)
        options_menu.addAction(self.toggle_lines_action)
        self.data_togglers_group.addAction(self.toggle_lines_action)
        self.toggle_lines_action.setChecked(True)

        self.toggle_smoothed_data_action = QAction("Show smoothed data", parent, checkable=True)
        self.toggle_smoothed_data_action.triggered.connect(parent.update_viewer)
        options_menu.addAction(self.toggle_smoothed_data_action)

        self.toggle_aggregate_before_export = QAction("Aggregate before export", parent, checkable=True)
        self.toggle_aggregate_before_export.setChecked(True)
        options_menu.addAction(self.toggle_aggregate_before_export)

        self.toggle_skip_survey = QAction("Skip survey spectra", parent, checkable=True)
        self.toggle_skip_survey.setChecked(True)
        options_menu.addAction(self.toggle_skip_survey)

        save_logs_action = QAction("Print logs", parent)
        save_logs_action.triggered.connect(parent.save_logs)
        self.addAction(save_logs_action)

class Sidebars():
    def __init__(self, parent, workspace, logger):
        self.logger = logger

        self.parent = parent
        self.workspace = workspace
        
        self.current_spectrum = None
        self.current_region = None
        self.analysis_window = None

        self.init_right_panel()
        self.init_left_panel()

    def init_left_panel(self):
        self.logger.debug("Initializing left panel")
        left_panel = QWidget()
        self.left_panel = left_panel
        left_panel_layout = QVBoxLayout()

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search...")
        self.search_box.textChanged.connect(self.filter_tree)
        left_panel_layout.addWidget(self.search_box)

        self.spectra_tree = QTreeWidget()
        self.spectra_tree.itemClicked.connect(self.set_currents_spectrum)
        self.spectra_tree.itemClicked.connect(self.update_region_list)
        self.spectra_tree.itemClicked.connect(self.parent.update_viewer)
        # self.spectra_tree.currentItemChanged.connect(self.set_currents_spectrum)
        # self.spectra_tree.currentItemChanged.connect(self.update_region_list)
        # self.spectra_tree.currentItemChanged.connect(self.parent.update_viewer)
        self.spectra_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.spectra_tree.setHeaderHidden(True)
        self.spectra_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.spectra_tree.customContextMenuRequested.connect(self.show_context_menu)
        left_panel_layout.addWidget(self.spectra_tree)
        self.update_spectra_tree()

        automatic_analysis_button = QPushButton("Automatic analysis")
        automatic_analysis_button.clicked.connect(self.automatic_analysis)
        left_panel_layout.addWidget(automatic_analysis_button)

        analysis_button = QPushButton("Trend analysis")
        analysis_button.clicked.connect(self.open_analysis_window)
        left_panel_layout.addWidget(analysis_button)

        left_panel.setLayout(left_panel_layout)
    
    def set_currents_spectrum(self, item, column):
        self.logger.debug(f"Setting current spectrum")
        if item is not None:
            self.current_spectrum = item.data(0, Qt.UserRole)
            self.logger.debug(f"Current spectrum set to {self.current_spectrum}")
            self.current_region = None
    
    def set_current_region(self, item):
        self.logger.debug(f"Setting current region")
        if item is not None:
            self.current_region = item.data(Qt.UserRole)
            self.logger.debug(f"Current region set to {self.current_region}")
            self.set_cursors()

    def aggregate_left_panel_items(self):
        self.logger.debug("Aggregating left panel items")
        selected_items = self.spectra_tree.selectedItems()  # Get all selected items

        if not selected_items:
            return
        
        groups = [item for item in selected_items if not item.parent()]
        spectra = [item for item in selected_items if item.parent()]

        return groups, spectra
    
    def show_context_menu(self, position):
        """Shows a right-click menu with rename & delete options for multiple items."""
        self.logger.debug("Showing context menu")
        selected_items = self.aggregate_left_panel_items()

        if not selected_items:
            return  # No item was clicked

        groups, spectra = selected_items

        left_panel = self.left_panel
        menu = QMenu(left_panel)

        # If multiple groups are selected, show "Merge Groups"
        if len(groups) > 1 and not spectra:
            merge_action = QAction("Merge Groups", left_panel)
            merge_action.triggered.connect(lambda: self.merge_selected_groups(groups))
            menu.addAction(merge_action)
        # If multiple items are selected, show "Delete"
        if len(groups) + len(spectra) > 1:
            delete_action = QAction("Delete", left_panel)
            delete_action.triggered.connect(lambda: self.delete_selected_items(groups + spectra))
            menu.addAction(delete_action)
        else:
            selected_items = groups + spectra
            item = selected_items[0]
            if not item.parent():
                rename_action = QAction("Rename", left_panel)
                rename_action.triggered.connect(lambda: self.rename_group(item))
                menu.addAction(rename_action)
                delete_action = QAction("Delete", left_panel)
                delete_action.triggered.connect(lambda: self.delete_group(item))
                menu.addAction(delete_action)
            else:
                rename_action = QAction("Rename", left_panel)
                rename_action.triggered.connect(lambda: self.rename_spectra(item))
                menu.addAction(rename_action)
                delete_action = QAction("Delete", left_panel)
                delete_action.triggered.connect(lambda: self.delete_spectrum(item))
                menu.addAction(delete_action)

        menu.exec(self.spectra_tree.viewport().mapToGlobal(position))

    def merge_selected_groups(self, items):
        """Merges multiple selected groups into one."""
        self.logger.debug("Merging selected groups")
        # Get names of selected groups
        group_names = [item.text(0) for item in items]
        # Ask user for the new group name
        new_group_name, ok = QInputDialog.getText(self.left_panel, "Merge Groups", "Enter new group name:", text="New Group")
        if not ok or not new_group_name:
            return  # User canceled
        if new_group_name in self.workspace.groups:
            QMessageBox.warning(self.left_panel, "Merge Error", "A group with this name already exists.")
            return
        self.workspace.merge_groups(new_group_name, group_names)
        # Remove old groups from UI
        for item in items:
            index = self.spectra_tree.indexOfTopLevelItem(item)
            self.spectra_tree.takeTopLevelItem(index)
        # Add new merged group to UI
        new_group_item = QTreeWidgetItem([new_group_name])
        self.spectra_tree.addTopLevelItem(new_group_item)
        # Add merged spectra to new group in UI
        for spectrum in self.workspace.groups[new_group_name]:
            spectrum_item = QTreeWidgetItem([spectrum.name])
            spectrum_item.setData(0, Qt.UserRole, spectrum)
            new_group_item.addChild(spectrum_item)

    def delete_selected_items(self, items):
        self.logger.debug("Deleting selected items")
        confirm = QMessageBox.question(
            self.left_panel, "Delete Selected Items",
            f"Are you sure you want to delete {len(items)} selected items?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            groups_to_delete = []
            spectra_to_delete = []
            for item in items:
                if not item.parent():
                    groups_to_delete.append(item)
                else:
                    spectra_to_delete.append(item)
            for spectrum_item in spectra_to_delete:
                self.delete_spectrum(spectrum_item, with_dialog=False)
            for group_item in groups_to_delete:
                self.delete_group(group_item, with_dialog=False)

    def rename_group(self, item):
        self.logger.debug("Renaming group")
        new_name, ok = QInputDialog.getText(self.left_panel, "Rename Group", "Enter new group name:")
        if ok and new_name:
            self.workspace.rename_group(item.text(0), new_name)
            item.setText(0, new_name)

    def delete_group(self, item, with_dialog=True):
        self.logger.debug("Deleting group")
        group_name = item.text(0)
        if with_dialog:
            confirm = QMessageBox.question(
                self.left_panel, "Delete Group", f"Are you sure you want to delete {group_name} and all its spectra?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if not with_dialog or confirm == QMessageBox.Yes:
            self.workspace.delete_group(group_name)
            index = self.spectra_tree.indexOfTopLevelItem(item)
            self.spectra_tree.takeTopLevelItem(index)  # Remove from UI

    def rename_spectra(self, item):
        self.logger.debug("Renaming spectrum")
        spectrum = item.data(0, Qt.UserRole)  # Retrieve an object
        if not spectrum:
            return
        new_name, ok = QInputDialog.getText(self.left_panel, "Rename Spectrum", "Enter new name:", text=spectrum.name)
        if ok and new_name:
            spectrum.name = new_name  # Update an object
            item.setText(0, new_name)  # Update UI

    def delete_spectrum(self, item, with_dialog=True):
        self.logger.debug("Deleting spectrum")
        parent_item = item.parent()
        if not parent_item:
            return
        spectrum = item.data(0, Qt.UserRole)  # Retrieve an object
        group_name = parent_item.text(0)
        if with_dialog:
            confirm = QMessageBox.question(
                self.left_panel, "Delete Spectrum", f"Are you sure you want to delete {spectrum.name}?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if not with_dialog or confirm == QMessageBox.Yes:
            idx = self.workspace.groups[group_name].index(spectrum)
            self.workspace.delete_spectrum(group_name, idx)
            parent_item.removeChild(item)  # Remove from UI

    def update_spectra_tree(self):
        self.logger.debug("Updating spectra tree")
        self.spectra_tree.clear()
        tree = {}
        # construct tree
        for s in self.workspace.spectra:
            file = s.file
            group = s.group
            spectrum_item = QTreeWidgetItem([s.name])
            QTreeWidgetItem.setExpanded
            spectrum_item.setData(0, Qt.UserRole, s)
            if file is None:
                file = "Unsorted"
            if group is None:
                group = "Unsorted"
            if file not in tree:
                file_item = QTreeWidgetItem([file])
                tree[file] = (file_item, {})
                self.spectra_tree.addTopLevelItem(file_item)
            if group not in tree[file][1]:
                group_item = QTreeWidgetItem([group])
                tree[file][1][group] = group_item
                tree[file][0].addChild(group_item)
            tree[file][1][group].addChild(spectrum_item)
        self.spectra_tree.expandAll()

    def filter_tree(self, text):
        text = text.lower()
        for i in range(self.spectra_tree.topLevelItemCount()):
            file_item = self.spectra_tree.topLevelItem(i)
            self.filter_item(file_item, text)
    
    def filter_item(self, item, text):
        match = text in item.text(0).lower()

        if match:
            # If the item matches, show it and all its children
            item.setHidden(False)
            for i in range(item.childCount()):
                child = item.child(i)
                self.show_all_children(child)
            return True
        else:
            # If item doesn't match, check if any children match
            child_match = False
            for i in range(item.childCount()):
                child = item.child(i)
                if self.filter_item(child, text):
                    child_match = True

            item.setHidden(not child_match)
            return child_match

    def show_all_children(self, item):
        item.setHidden(False)
        for i in range(item.childCount()):
            self.show_all_children(item.child(i))

    def get_selected_spectra(self, skip_survey=True):
        self.logger.debug("Getting selected spectra")
        selected_items = self.aggregate_left_panel_items()
        if not selected_items:
            return

        groups, spectra = selected_items
        spectra_list = []
        for group in groups:
            spectra_list.extend(self.workspace.groups[group.text(0)])
        for spectrum_item in spectra:
            spectrum = spectrum_item.data(0, Qt.UserRole)
            if spectrum not in spectra_list:
                spectra_list.append(spectrum)

        if skip_survey:
            spectra_list = [s for s in spectra_list if not s.is_survey]

        return spectra_list
    
    def predict(self):
        self.logger.debug("Predicting")
        skip_survey = self.parent.toolbar.toggle_skip_survey.isChecked()
        spectra_list = self.get_selected_spectra(skip_survey)
        if spectra_list is None or len(spectra_list) == 0:
            spectra_list = self.workspace.aggregate_spectra()
        self.workspace.predict(spectra=spectra_list)
        self.parent.toolbar.toggle_labeled_data_action.setChecked(True)
        self.parent.update_viewer()

    def post_process(self):
        self.logger.debug("Post processing")
        skip_survey = self.parent.toolbar.toggle_skip_survey.isChecked()
        spectra_list = self.get_selected_spectra(skip_survey)
        if spectra_list is None or len(spectra_list) == 0:
            spectra_list = self.workspace.aggregate_spectra()
        spectra_list = [s for s in spectra_list if not s.is_analyzed and s.is_predicted]
        progress_window = ProgressBarWindow(self.workspace.post_process, len(spectra_list), spectra_list)
        progress_window.exec()

        self.parent.toolbar.toggle_lines_action.setChecked(True)
        self.update_region_list()
        self.update_analysis_window()
        self.parent.update_viewer()
    
    def automatic_analysis(self):
        self.logger.debug("Automatic analysis")
        skip_survey = self.parent.toolbar.toggle_skip_survey.isChecked()
        spectra_list = self.get_selected_spectra(skip_survey)
        if spectra_list is None or len(spectra_list) == 0:
            spectra_list = self.workspace.aggregate_spectra()
        self.workspace.predict(spectra=spectra_list)
        spectra_list = [s for s in spectra_list if not s.is_analyzed and s.is_predicted]

        progress_window = ProgressBarWindow(self.workspace.post_process, len(spectra_list), spectra_list)
        progress_window.exec()

        self.parent.toolbar.toggle_lines_action.setChecked(True)
        self.update_region_list()
        self.update_analysis_window()
        self.parent.update_viewer()
    
    def open_analysis_window(self):
        self.logger.debug("Opening analysis window")
        self.analysis_window = AnalysisWindow(self.workspace)
        self.analysis_window.show()

    #TODO: updating window on change
    def update_analysis_window(self):
        if self.analysis_window is not None and self.analysis_window.isVisible():
            self.logger.debug("Updating analysis window")
            self.analysis_window.update_lists_layout()

    def init_right_panel(self):
        self.logger.debug("Initializing right panel")
        right_panel = QWidget()
        self.right_panel = right_panel
        right_panel_layout = QVBoxLayout()

        # Region parameters
        region_label = QLabel("Optimization tools")
        right_panel_layout.addWidget(region_label)

        # Region list
        self.region_list = QListWidget()
        self.region_list.setFixedHeight(100)
        self.region_list.itemClicked.connect(self.set_current_region)
        self.region_list.itemClicked.connect(self.load_region_tab)
        # self.region_list.currentItemChanged.connect(self.set_current_region)
        # self.region_list.currentItemChanged.connect(self.load_region_tab)
        right_panel_layout.addWidget(self.region_list)

        refit_layout = QHBoxLayout()
        self.refit_region_button = QPushButton("Optimize")
        self.refit_region_button.clicked.connect(self.refit_region)
        self.refit_region_button.clicked.connect(self.update_lines_settings_tab)
        # self.refit_region_button.clicked.connect(self.update_analysis_window)
        self.refit_region_button.clicked.connect(self.parent.update_viewer)
        refit_layout.addWidget(self.refit_region_button)

        self.reoptimize_all_box = QCheckBox("Reoptimize all")
        self.reoptimize_all_box.setChecked(False)
        refit_layout.addWidget(self.reoptimize_all_box)

        self.fast_fit = QCheckBox("Fast fit")
        self.fast_fit.setChecked(True)
        refit_layout.addWidget(self.fast_fit)

        right_panel_layout.addLayout(refit_layout)

        create_delete_layout = QHBoxLayout()
        self.create_region_button = QPushButton("Create Region")
        self.create_region_button.clicked.connect(self.create_new_region)
        create_delete_layout.addWidget(self.create_region_button)

        delete_button = QPushButton("Delete Region")
        delete_button.clicked.connect(self.delete_region)
        delete_button.clicked.connect(self.parent.update_viewer)
        create_delete_layout.addWidget(delete_button)

        right_panel_layout.addLayout(create_delete_layout)

        self.region_tabs = QTabWidget()
        right_panel_layout.addWidget(self.region_tabs)
        self.create_region_tabs()

        right_panel.setLayout(right_panel_layout)

    def set_cursors(self):
        self.logger.debug("Setting cursors in right panel")
        if self.current_region is not None:
            self.parent.update_cursors(self.current_region)

    def update_region_list(self):
        self.region_list.clear()
        spectrum = self.current_spectrum
        self.logger.debug(f"Updating region list for {spectrum}")
        if spectrum is not None and len(spectrum.regions) > 0:
            for region in spectrum.regions:
                item = QListWidgetItem(f"Region {spectrum.regions.index(region)}")
                item.setData(Qt.UserRole, region)
                self.region_list.addItem(item)
            if len(spectrum.regions) > 0:
                self.current_region = spectrum.regions[0]
                self.region_list.setCurrentRow(0)
            self.load_region_tab()

    def create_region_tabs(self):
        """
        Creates the tabs for the right panel of the window.
        The tabs are Region settings and Peaks settings.
        """
        self.logger.debug("Creating region tabs")
        region_debug_tab = QWidget()
        tab_layout = QFormLayout()
        region_debug_tab.setLayout(tab_layout)
        self.region_tabs.addTab(region_debug_tab, "Region settings")

        lines_debug_tab = ScrollableWidget()
        add_line_button = QPushButton("Add peak")
        add_line_button.clicked.connect(self.add_line)
        lines_debug_tab.layout().addWidget(add_line_button)
        self.region_tabs.addTab(lines_debug_tab, "Peaks settings")

    def load_region_tab(self):
        self.logger.debug("Loading region tab")
        self.clear_tabs()
        if self.current_region is not None:
            self.load_region_settings_tab()
            self.fixed_params_cb = []
            self.load_lines_settings_tab()
    
    def clear_tabs(self):
        self.logger.debug("Clearing tabs")
        self.clear_layout(self.region_tabs.widget(0).layout())
        self.clear_layout(self.region_tabs.widget(1).content_layout)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)  # Remove item
            widget = item.widget()
            if widget:
                widget.deleteLater()  # Delete the widget properly

    def load_region_settings_tab(self):
        self.logger.debug("Loading region settings tab")
        region = self.current_region
        tab_layout = self.region_tabs.widget(0).layout()

        start_param = self.create_region_param_input(region, 'start_point') 
        tab_layout.addRow('From (eV):', start_param)

        end_param = self.create_region_param_input(region, 'end_point')
        tab_layout.addRow('To (eV):', end_param)
    
    def update_region_settings_tab(self):
        self.logger.debug("Updating region settings tab")
        tab_widget = self.region_tabs.widget(0)
        region = self.current_region
        for param, param_input in zip(('start_point', 'end_point'), tab_widget.findChildren(QLineEdit)):
            param_input.setText(f"{getattr(region, param):.2f}")

    def create_region_param_input(self, region, param):
        self.logger.debug("Creating region parameter input")
        current_value = getattr(region, param)
        if not isinstance(current_value, str):
            current_value = f"{current_value:.2f}"
        input_edit = QLineEdit(current_value)
        input_edit.editingFinished.connect(lambda: self.update_region_param(region, param, input_edit.text().strip(), input_edit))
        input_edit.editingFinished.connect(self.parent.update_viewer)
        return input_edit

    def update_region_param(self, region, param, value, edit):
        self.logger.debug(f"Updating region parameter {param} to {value}")
        edit.setText(value)
        self.workspace.change_region_parameter(region, self.current_spectrum, param, value)

    def add_line_to_tab(self, line):
        self.logger.debug("Adding peak to tab")

        region = self.current_region
        tab_layout = self.region_tabs.widget(1).content_layout
        # ((parameter1_label, obj parameter), ...)
        editable_params = (
            ('Position', 'loc'),
            ('FWHM', 'fwhm'),
            ('Amplitude', 'const'),
            ('GL ratio', 'gl_ratio'),
        )
        noneditable_params = (
            ('Area', 'area'),
            ('Height', 'height')
        )
        line_group = QGroupBox(f"Peak {region.lines.index(line)}")
        line_layout = QFormLayout()
        line_group.setLayout(line_layout)
        cb_layout = QHBoxLayout()
    
        cb_label = QLabel("Fix parameters")
        cb_layout.addWidget(cb_label)
        cb_layout.setAlignment(Qt.AlignRight)
        line_layout.addRow(cb_layout)
        cb_list = []
        for (param_label, param) in editable_params:
            layout = QHBoxLayout()
            cb = QCheckBox()
            #TODO: set constraints button
            # cb.stateChanged.connect()
            param_input = self.create_line_param_input(line, param)
            cb_list.append(cb)
            layout.addWidget(param_input)
            layout.addWidget(cb)
            line_layout.addRow(param_label, layout)
        for (param_label, param) in noneditable_params:
            line_layout.addRow(param_label, self.create_line_param_input(line, param, read_only=True))

        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(lambda: self.delete_line(region.lines.index(line)))
        self.fixed_params_cb.append(cb_list)
        line_layout.addRow(delete_button)
        tab_layout.addWidget(line_group)

    #TODO: set constraints button
    # def fix_unfix_line_parameter(self, line, parameter):
    #     pass

    def load_lines_settings_tab(self):
        self.logger.debug("Loading peaks settings tab")
        region = self.current_region
        for line in region.lines:
            self.add_line_to_tab(line)

    def update_lines_settings_tab(self):
        self.logger.debug("Updating lines settings tab")
        tab = self.region_tabs.widget(1)
        region = self.current_region
        for line, line_setting in zip(region.lines, tab.findChildren(QGroupBox)):
            line_setting.setTitle(f"Peak {region.lines.index(line)}")
            for param, param_input in zip(('loc', 'fwhm', 'const', 'gl_ratio', 'area', 'height'), line_setting.findChildren(QLineEdit)):
                param_input.setText(f"{getattr(line, param):.2f}")

    def remove_line_settings(self, line_idx):
        self.logger.debug("Removing peak settings")
        tab_layout = self.region_tabs.widget(1).content_layout
        widget = tab_layout.itemAt(line_idx).widget()
        tab_layout.removeWidget(widget)
        widget.deleteLater()
        self.fixed_params_cb.pop(line_idx)

    def delete_line(self, line_idx):
        self.logger.debug("Deleting peak")
        self.workspace.delete_line(self.current_region, line_idx)
        self.remove_line_settings(line_idx)
        self.parent.update_viewer()

    def create_line_param_input(self, line, param, read_only=False):
        current_value = getattr(line, param)
        if not isinstance(current_value, str):
            current_value = f"{current_value:.3f}"
        input_edit = QLineEdit(current_value)
        if read_only:
            input_edit.setReadOnly(True)
        else:
            input_edit.editingFinished.connect(lambda: self.update_line_param(line, param, input_edit.text().strip(), input_edit))
            input_edit.editingFinished.connect(self.parent.update_viewer)
        return input_edit

    def update_line_param(self, line, param, value, edit):
        self.logger.debug(f"Updating peak parameter {param} to {value}")
        edit.setText(value)
        value = float(value)
        self.workspace.change_line_parameter(line, param, value)

    def delete_region(self):
        self.logger.debug("Deleting region")
        selected_item = self.region_list.currentItem()
        if selected_item is not None:
            region = selected_item.data(Qt.UserRole)
            self.workspace.delete_region(region, self.current_spectrum)
            self.region_list.takeItem(self.region_list.row(selected_item))
            self.region_list.setCurrentRow(self.region_list.count() - 1)

    def create_new_region(self):
        self.logger.debug("Creating new region")
        spectrum = self.current_spectrum
        if spectrum is not None:
            x1 = spectrum.x[0]
            x2 = spectrum.x[-1]
            region = self.workspace.create_new_region(x1, x2, spectrum=spectrum)
            item = QListWidgetItem(f"Region {spectrum.regions.index(region)}")
            item.setData(Qt.UserRole, region)
            self.region_list.addItem(item)
            self.region_list.setCurrentRow(self.region_list.count() - 1)

    #TODO: adding lines by maxima fitting errors
    def add_line(self):
        self.logger.debug("Adding peak button clicked")
        region = self.current_region
        if region is not None:
            if len(region.lines) != 0:
                self.workspace.add_line(
                    region, self.current_region.lines[-1].loc + 1, self.current_region.lines[-1].scale, self.current_region.lines[-1].const, self.current_region.lines[-1].gl_ratio
                )
            else:
                self.workspace.add_line(region)
            self.add_line_to_tab(self.current_region.lines[-1])
            self.parent.update_viewer()

    def refit_region(self):
        self.logger.debug("Refitting region")
        selected_item = self.region_list.currentItem()
        if selected_item is None:
            return
        if self.fast_fit.isChecked():
            fit_alg = 'least squares'
        else:
            fit_alg = 'differential evolution'
        region = selected_item.data(Qt.UserRole)
        if self.reoptimize_all_box.isChecked():
            self.workspace.refit(region, tol=0.15, full_refit=True, fit_alg=fit_alg)
        else:
            fixed_params = self.parse_fixed_params()
            self.workspace.refit(region, fixed_params=fixed_params, tol=0.2, fit_alg=fit_alg, loc_tol=1)

    def parse_fixed_params(self):
        return [i for i, box in enumerate(chain(*self.fixed_params_cb)) if box.isChecked()]

class AnalysisWindow(QDialog):
    def __init__(self, workspace):
        super().__init__()
        self.setWindowTitle("Analysis Window")
        self.setGeometry(100, 100, 700, 400)
        self.workspace = workspace

        self.attrs = ('loc', 'area', 'fwhm', 'gl_ratio')
        self.header = ('Position', 'Area', 'FWHM', 'GL')
        self.fmt = ('{:.1f}', '{:.1f}', '{:.1f}', '{:.2f}')

        layout = QVBoxLayout()

        # Initial tree widget list
        self.tree_widget = QTreeWidget()
        self.tree_widget.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tree_widget.setHeaderHidden(True)
        self.populate_tree()
        tree_layout = QVBoxLayout()
        tree_layout.addWidget(QLabel("Available Objects:"))
        tree_layout.addWidget(self.tree_widget)

        # Second list widget
        self.table_widget = QTableWidget()
        self.selected_objects = []
        self.selected_objects_attrs = []
        self.table_widget.setColumnCount(len(self.header))
        self.table_widget.setHorizontalHeaderLabels(self.header)
        self.table_widget.setSelectionBehavior(QTableWidget.SelectRows)  # Select full rows
        self.table_widget.setSelectionMode(QTableWidget.ExtendedSelection)
        total_width = sum(self.table_widget.columnWidth(i) for i in range(self.table_widget.columnCount())) + 2
        self.table_widget.setMinimumSize(total_width, 300)
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        list_layout = QVBoxLayout()
        list_layout.addWidget(QLabel("Selected Objects:"))
        list_layout.addWidget(self.table_widget)

        # Buttons to move items
        btn_layout = QVBoxLayout()

        btn_add = QPushButton("→")
        btn_add.clicked.connect(self.add_items)
        btn_add.setFixedSize(30, 20)

        btn_update = QPushButton("↻")
        btn_update.setFixedSize(30, 20)
        btn_update.clicked.connect(self.update_lists_layout)

        btn_remove = QPushButton("←")
        btn_remove.clicked.connect(self.remove_items)
        btn_remove.setFixedSize(30, 20)

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_update)
        btn_layout.addWidget(btn_remove)

        # Dropdown for analysis options
        self.parameter_option = QComboBox()
        self.options = ['Position', 'Area', 'FWHM', 'GL']
        self.parameter_option.addItems(self.options)
        tree_layout.addWidget(QLabel("Select parameter to view:"))
        tree_layout.addWidget(self.parameter_option)

        # Layout for lists and buttons
        lists_layout = QHBoxLayout()
        lists_layout.addLayout(tree_layout)
        lists_layout.addLayout(btn_layout)
        lists_layout.addLayout(list_layout)

        # Buttons for actions
        btn_layout_bottom = QHBoxLayout()

        btn_proceed = QPushButton("Visualize")
        btn_proceed.clicked.connect(self.view)
        btn_layout_bottom.addWidget(btn_proceed)

        # Add widgets to layout
        layout.addLayout(lists_layout)
        layout.addLayout(btn_layout_bottom)

        self.setLayout(layout)

    def populate_tree(self):
        self.tree_widget.clear()
        for group, spectra in self.workspace.groups.items():
            group_item = QTreeWidgetItem([group])
            group_item.setFlags(group_item.flags() | Qt.ItemIsDropEnabled)
            for spectrum in spectra:
                if not spectrum.is_analyzed:
                    continue
                spectrum_item = QTreeWidgetItem([spectrum.name])
                for reg_n, region in enumerate(spectrum.regions):
                    region_item = QTreeWidgetItem([f"Region {reg_n}"])
                    for num, line in enumerate(region.lines):
                        line_name = f"Peak {num} at {line.loc:.1f}"
                        line_item = QTreeWidgetItem([line_name])
                        line_item.setData(0, Qt.UserRole, line)
                        region_item.addChild(line_item)
                    spectrum_item.addChild(region_item)
                group_item.addChild(spectrum_item)
            if group_item.childCount() > 0:
                self.tree_widget.addTopLevelItem(group_item)
        self.tree_widget.expandAll()
    
    def traverse(self, item):
        if item.childCount() > 0:
            l = []
            for i in range(item.childCount()):
                l.extend(self.traverse(item.child(i)))
            return l
        else:
            return [item]

    def add_items(self):
        """Move selected objects from tree to list, preventing duplicates."""
        selected_items = self.tree_widget.selectedItems()
        selected_lines = []
        for item in selected_items:
            selected_lines.extend(self.traverse(item))
        for item in selected_lines:
            if item.data(0, Qt.UserRole) not in self.selected_objects:
                self.selected_objects.append(item.data(0, Qt.UserRole))
        self.update_table()
    
    def update_attrs(self):
        self.selected_objects_attrs = []
        for item in self.selected_objects:
            params = [fmt.format(getattr(item, attr)) for attr, fmt in zip(self.attrs, self.fmt)]
            self.selected_objects_attrs.append(params)

    def update_table(self):
        self.table_widget.setRowCount(0)
        self.update_attrs()
        for line in self.selected_objects_attrs:
            self.append_row(line)
    
    def append_row(self, iterable):
        row_count = self.table_widget.rowCount()
        new_row = row_count
        self.table_widget.insertRow(new_row)
        for i, item in enumerate(iterable):
            self.table_widget.setItem(new_row, i, QTableWidgetItem(str(item)))
    
    def update_lists_layout(self):
        self.populate_tree()
        self.update_table()

    def remove_items(self):
        """Remove selected objects from the table."""
        selected_rows = sorted(set(index.row() for index in self.table_widget.selectedIndexes()), reverse=True)
        for row in selected_rows:
            self.table_widget.removeRow(row)
            self.selected_objects.pop(row)
            self.selected_objects_attrs.pop(row)

    def view(self):
        self.plot_dialog = QDialog()
        self.plot_dialog.setWindowTitle("Trend Plot")
        self.plot_dialog.setGeometry(100, 100, 800, 600)
        plot_layout = QVBoxLayout()
        self.plot_dialog.setLayout(plot_layout)
        plot_widget = pg.PlotWidget()
        plot_layout.addWidget(plot_widget)

        vb = plot_widget.getViewBox()
        for action in vb.menu.actions():
            action.setVisible(False)
        pi = plot_widget.getPlotItem()
        pi.ctrlMenu.menuAction().setVisible(False)

        color = self.palette().color(QtGui.QPalette.Base)
        plot_widget.setBackground(color)

        selected_option = self.parameter_option.currentText()
        y = self.workspace.build_trend(self.selected_objects, selected_option)
        x = np.arange(1, len(y) + 1)
        plot_widget.plot(x, y, pen={'width': 2, 'color': 'k'}, symbol='o', symbolPen={'color': 'k'}, symbolBrush=(0, 0, 0))

        self.plot_dialog.show()

class WorkerThread(QThread):
    progress_signal = Signal(int)  # Signal to update progress bar
    finished_signal = Signal()  # Signal to close progress window

    def __init__(self, generator_func, *args, **kwargs):
        super().__init__()
        self.generator_func = generator_func
        self.running = True  # Control flag for stopping
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Run the generator function and emit progress updates"""
        for i, output in enumerate(self.generator_func(*self.args, **self.kwargs)):
            if not self.running:  # Stop if interrupted
                break
            self.progress_signal.emit(i)  # Send progress to UI
        
        self.finished_signal.emit()  # Notify that process is done

    def stop(self):
        """Stop the thread gracefully"""
        self.running = False  # Set flag to stop loop

class ProgressBarWindow(QDialog):
    """Popup window that displays progress bar based on a generator"""
    def __init__(self, generator_func, iterations, *func_args):
        super().__init__()

        self.setWindowTitle("Processing...")
        self.setFixedSize(300, 150)

        # Layout
        self.layout = QVBoxLayout(self)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, iterations)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        # Stop Button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.layout.addWidget(self.stop_button)

        # Worker Thread
        self.worker_thread = WorkerThread(generator_func, *func_args)
        self.worker_thread.progress_signal.connect(self.update_progress)  # Connect progress update
        self.worker_thread.finished_signal.connect(self.close)  # Close when finished
        self.worker_thread.start()  # Start processing

    def update_progress(self, value):
        """Update progress bar when receiving a new value from generator"""
        self.progress_bar.setValue(value)

    def stop_processing(self):
        """Stop the generator function"""
        self.worker_thread.stop()  # Stop thread
        self.worker_thread.quit()  # Ensure it exits
        self.worker_thread.wait()  # Wait for cleanup
        self.close()  # Close the progress window

class ScrollableWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Create main layout
        layout = QVBoxLayout(self)

        # Create a Scroll Area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)  # Allows resizing of the contained widget

        # Create a container widget for the scroll area
        content_widget = QWidget()
        self.content_layout = QVBoxLayout(content_widget)

        # Set the container widget as the scroll area's widget
        self.scroll_area.setWidget(content_widget)

        # Add the scroll area to the main layout
        layout.addWidget(self.scroll_area)

        self.setLayout(layout)
        # self.resize(300, 400)  # Set window size

class PlotCanvas(pg.PlotWidget):
    # colors = [
    #     'darkblue',
    #     'crimson',
    #     'khaki',
    #     'orange',
    #     'lightgreen',
    #     'magenta',
    #     'lightpink',
    #     'deepskyblue'
    # ]

    def __init__(self, parent, workspace):        
        super().__init__(parent)

        self.parent = parent
        self.workspace = workspace

        self.showGrid(x=True, y=True)
        self.setMouseEnabled(x=True, y=True)
        self.setLabel("bottom", "Binding Energy (eV)")
        vb = self.getViewBox()
        # vb.setMouseMode(pg.ViewBox.RectMode)
        vb.setMenuEnabled(False)

        self.c1 = self.create_cursor('start_point')
        self.c2 = self.create_cursor('end_point')
        self.cursor_pen = {'color': 'r', 'width': 2, 'style': Qt.DashLine}

        self.mask_parameters = ((0, 0, 255, 100), {255, 0, 0, 255})

    def reload_spectrum(self, spectrum):
        self.spectrum = spectrum
        self.setTitle(spectrum.name)

        x, y = spectrum.x, spectrum.y
        y_smooth = spectrum.y_smoothed
        regions = spectrum.regions

        self.main_curves = []
        self.regions_lines = []

        self.main_curves.extend([
            pg.PlotDataItem(x, y, pen={'color': 'k', 'width': 2}),
            pg.PlotDataItem(x, y_smooth, pen={'color': 'k', 'width': 2})
        ])

        self.create_masks()

        for region in regions:
            self.add_region(region)
    
    def _prepare_mask(self, mask):
        mask = mask.copy()
        v = np.lib.stride_tricks.sliding_window_view(mask, 3, writeable=True)
        f = np.array([0, 1, 0])
        v[(v == f).all(axis=1)] = np.array([[1, 1, 1]])
        return mask
    
    def create_masks(self):
        peak_mask, max_mask = self.spectrum.get_masks()
        if peak_mask is None or max_mask is None:
            return
        
        peak_mask = self._prepare_mask(peak_mask)
        max_mask = self._prepare_mask(max_mask)

        x = self.spectrum.x_interpolated
        y = self.spectrum.y_interpolated
        
        curve = pg.PlotDataItem(x, y, pen={'color': 'k', 'width': 2})
        min_to_fill = np.zeros_like(x)

        self.main_curves.append(curve)

        if peak_mask.any():
            c1 = np.where(peak_mask, y, np.nan)
            c2 = np.where(peak_mask, min_to_fill, np.nan)
            curve_peak_1 = pg.PlotDataItem(x, c1, pen=self.mask_parameters[0])
            curve_peak_2 = pg.PlotDataItem(x, c2, pen=self.mask_parameters[0])
            fill_peak = pg.FillBetweenItem(curve_peak_1, curve_peak_2, brush=self.mask_parameters[0])
            self.main_curves.append(fill_peak)

        if max_mask.any():
            c1 = np.where(max_mask, y, np.nan)
            c2 = np.where(max_mask, min_to_fill, np.nan)
            curve_max_1 = pg.PlotDataItem(x, c1, pen=self.mask_parameters[1])
            curve_max_2 = pg.PlotDataItem(x, c2, pen=self.mask_parameters[1])
            fill_max = pg.FillBetweenItem(curve_max_1, curve_max_2, brush=self.mask_parameters[1])
            self.main_curves.append(fill_max)
    
    def delete_region(self, region_idx):
        region = self.regions_lines.pop(region_idx)
        for line in region:
            self.removeItem(line)
    
    def add_region(self, region):
        region_curves = []

        reg_x, back, s, *reg_lines = region.draw_lines()
        region_curves.extend([
            pg.PlotDataItem(reg_x, back, pen={'color': 'k', 'width': 2, 'style': Qt.DashLine}),
            pg.PlotDataItem(reg_x, s, pen={'color': 'k', 'width': 2, 'style': Qt.DotLine})
        ])

        for i, line in enumerate(reg_lines):
            color = pg.mkColor((i, len(reg_lines)))
            region_curves.append(
                pg.PlotDataItem(reg_x, line, pen={'color': color, 'width': 3})
            )
        self.regions_lines.append(region_curves)
    
    # def delete_line(self, region_idx, line_idx):
    #     line = self.regions_lines[region_idx].pop(2 + line_idx)
    #     self.removeItem(line)
    
    # def add_line(self, region_idx):
    #     self.regions_lines[region_idx].append(pg.PlotDataItem(0, 0, pen=next(self.colors)))
    
    def update_data(self):
        for region_curves, region in zip(self.regions_lines, self.spectrum.regions):
            reg_x, back, s, *reg_lines = region.draw_lines()
            for line_curve, data in zip(region_curves, (back, s, *reg_lines)):
                line_curve.setData(reg_x, data)
    
    def change_smoothing_plotting(self, smoothed=False):
        vis = {'color': 'k', 'width': 2, 'alpha': 1}
        transp = {'color': 'k', 'width': 1, 'alpha': 0.2}

        if smoothed:
            self.main_curves[0].setPen(transp)
            self.main_curves[1].setPen(vis)
        else:
            self.main_curves[0].setPen(vis)
            self.main_curves[1].setPen(transp)

    def update_plot(self, disp_type='lines', smoothed=False):
        self.clear()
        if disp_type == 'lines':
            self.addItem(self.main_curves[0])
            self.change_smoothing_plotting(smoothed)
            if smoothed:
                self.addItem(self.main_curves[1])
            for region_curves in self.regions_lines:
                for curve in region_curves:
                    self.addItem(curve)

        elif disp_type == 'labeled':
            for curve in self.main_curves[2:]:
                self.addItem(curve)

        elif disp_type == 'raw':
            self.addItem(self.main_curves[0])
            self.change_smoothing_plotting(smoothed)
            if smoothed:
                self.addItem(self.main_curves[1])

    def load_cursors(self, region):
        self.region = region
        self.set_cursors(region.start_point, region.end_point)

    def create_cursor(self, param):
        c = pg.InfiniteLine(angle=90, movable=True, pen=None)
        c.sigPositionChangeFinished.connect(lambda cursor: self.update_position(cursor.value(), param))
        return c

    def update_cursors(self, pos1, pos2):
        self.c1.setPos(pos1)
        self.c2.setPos(pos2)

    def set_cursors(self, pos1, pos2):
        self.update_cursors(pos1, pos2)
        self.c1.setPen(self.cursor_pen)
        self.c2.setPen(self.cursor_pen)
        self.addItem(self.c1)
        self.addItem(self.c2)
    
    def update_position(self, val, param):
        if self.spectrum is not None:
            x1 = self.c1.value()
            x2 = self.c2.value()
            if x1 > x2:
                x1, x2 = x2, x1
                self.workspace.change_region_parameter(self.region, self.spectrum, 'start_point', x1)
                self.workspace.change_region_parameter(self.region, self.spectrum, 'end_point', x2)
            else:
                self.workspace.change_region_parameter(self.region, self.spectrum, param, val)
        
        self.update_data()
        self.parent.sidebars.update_region_settings_tab()


def main():
    app = QApplication(sys.argv)

    # Set the application palette to a light theme
    set_palette(app)

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


def set_palette(app):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))  # Light gray background
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))  # Black text
    palette.setColor(QPalette.Base, QColor(255, 255, 255))  # White base
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))  # Light gray alternate base
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))  # Light yellow tooltip base
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))  # Black tooltip text
    palette.setColor(QPalette.Text, QColor(0, 0, 0))  # Black text
    palette.setColor(QPalette.Button, QColor(240, 240, 240))  # Light gray button
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))  # Black button text
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))  # White bright text
    palette.setColor(QPalette.Link, QColor(0, 0, 255))  # Blue link
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))  # Light blue highlight
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))  # White highlighted text

    app.setPalette(palette)


if __name__ == "__main__":
    main()
