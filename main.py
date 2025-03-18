import sys
import io
import logging
import traceback
from itertools import chain

import numpy as np
import torch

from PySide6.QtWidgets import *
from PySide6.QtGui import QAction, QPalette, QColor, QActionGroup
from PySide6.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

from model.models.model_deeper import XPSModel
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
    
    def setup_logging(self, log_level=logging.INFO):
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
        self.logger.info("Logging initialized.")

        sys.excepthook = self.handle_unhandled_exception

    def handle_unhandled_exception(self, exctype, value, tb):
        """Log unhandled exceptions with full traceback."""
        error_message = "".join(traceback.format_exception(exctype, value, tb))
        self.logger.critical(f"Unhandled Exception:\n{error_message}")

    def load_model(self):
        self.logger.info("Loading model")
        m = XPSModel()
        m.load_state_dict(
            torch.load('model.pt', map_location=torch.device('cpu'), weights_only=True)
        )
        m.eval()
        return m

    def initUI(self):
        self.logger.info("Initializing UI")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Toolbar for actions
        self.toolbar = Toolbar(self)
        self.addToolBar(self.toolbar)

        # Splitter for sidebar, main content, and right panel
        splitter = QSplitter()

        self.main_content = self.main_content_widget()
        self.sidebars = Sidebars(self, self.workspace, self.logger)

        splitter.addWidget(self.sidebars.left_panel)
        splitter.addWidget(self.main_content)
        splitter.addWidget(self.sidebars.right_panel)

        # Set the splitter to the central widget
        central_layout = QHBoxLayout()
        central_layout.addWidget(splitter)
        central_widget.setLayout(central_layout)
    
    def main_content_widget(self):
        self.logger.info("Initializing main content widget")
        main_content_widget = QWidget()
        # Main content layout
        main_content_layout = QVBoxLayout()

        # Figure for plotting spectra
        self.canvas = PlotCanvas(self.workspace, self)
        self.navigation_toolbar = NavigationToolbar(self.canvas, self)
        main_content_layout.addWidget(self.navigation_toolbar)
        main_content_layout.addWidget(self.canvas)

        main_content_widget.setLayout(main_content_layout)

        return main_content_widget
    
    def update_viewer(self):
        spectrum = self.sidebars.current_spectrum
        region = self.sidebars.current_region
        if spectrum is not None:
            add_smoothing = self.toolbar.toggle_smoothed_data_action.isChecked()
            if spectrum.is_predicted and self.toolbar.toggle_labeled_data_action.isChecked():
                self.canvas.plot(spectrum, type='labeled')
            elif self.toolbar.toggle_raw_data_action.isChecked():
                self.canvas.plot(spectrum, type='raw', add_smoothing=add_smoothing)
            else:
                self.canvas.plot(spectrum, add_smoothing=add_smoothing)
        if region is not None:
            self.set_cursors(region, spectrum)
    
    def set_cursors(self, region, spectrum):
        self.canvas.load_cursors(region, spectrum)
    
    def update_sidebars(self):
        self.logger.info("Updating sidebars")
        self.sidebars.update_spectra_tree()
    
    def load_spectra(self):
        self.logger.info("Loading spectra")
        files, _ = QFileDialog.getOpenFileNames(self, "Load Spectra", "", "All Files (*);;Vamas Files (*.vms);;Text Files (*.txt);;SPECS Files (*.xml)")
        if files:
            self.workspace.load_files(*files)
            self.update_sidebars()

    def change_prediction_threshold(self):
        self.logger.info("Changing prediction threshold")
        value = self.workspace.pred_threshold
        new_threshold, ok = QInputDialog.getDouble(self, "Change Prediction Threshold", "Enter new prediction threshold:", value, 0, 1, 2)
        if ok:
            self.workspace.set_prediction_threshold(new_threshold)
        self.update_viewer()
    
    def save_logs(self):
        self.logger.info("Saving logs")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Log File", "", "Text Files (*.txt);;All Files (*)")

        if file_path:  # If user selects a file
            self.log_buffer.seek(0)  # Move to the start of the buffer
            log_contents = self.log_buffer.read()

            with open(file_path, "w") as f:
                f.write(log_contents)

            self.logger.info(f"All logs have been saved to {file_path}")

class Toolbar(QToolBar):
    def __init__(self, parent):
        super().__init__("Main Toolbar", parent)

        # Load Spectra Action
        load_action = QAction("Load Spectra", parent)
        load_action.triggered.connect(parent.load_spectra)
        self.addAction(load_action)

        # Options Menu
        options_menu = QMenu("Options...", parent)
        options_action = self.addAction("Options...")
        options_action.setMenu(options_menu)
        options_action.triggered.connect(lambda: options_menu.exec(self.mapToGlobal(self.rect().bottomLeft())))

        # Change Prediction Threshold Action
        change_threshold_action = QAction("Change Prediction Threshold", parent)
        change_threshold_action.triggered.connect(parent.change_prediction_threshold)
        options_menu.addAction(change_threshold_action)

        # Spectra Viewer Options Group
        self.data_togglers_group = QActionGroup(parent)
        self.data_togglers_group.setExclusive(True)

        # Toggle Labeled Data Action
        self.toggle_labeled_data_action = QAction("Show Labeled Data", parent, checkable=True)
        self.toggle_labeled_data_action.triggered.connect(parent.update_viewer)
        options_menu.addAction(self.toggle_labeled_data_action)
        self.data_togglers_group.addAction(self.toggle_labeled_data_action)

        # Toggle Raw Action
        self.toggle_raw_data_action = QAction("Show Raw Data", parent, checkable=True)
        self.toggle_raw_data_action.triggered.connect(parent.update_viewer)
        options_menu.addAction(self.toggle_raw_data_action)
        self.data_togglers_group.addAction(self.toggle_raw_data_action)

        # Toggle Line Data Action
        self.toggle_lines_action = QAction("Show Lines", parent, checkable=True)
        self.toggle_lines_action.triggered.connect(parent.update_viewer)
        options_menu.addAction(self.toggle_lines_action)
        self.data_togglers_group.addAction(self.toggle_lines_action)
        self.toggle_lines_action.setChecked(True)

        self.toggle_smoothed_data_action = QAction("Show Smoothed Data", parent, checkable=True)
        self.toggle_smoothed_data_action.triggered.connect(parent.update_viewer)
        options_menu.addAction(self.toggle_smoothed_data_action)

        save_logs_action = QAction("Print logs", parent)
        save_logs_action.triggered.connect(parent.save_logs)
        self.addAction(save_logs_action)

class Sidebars():
    def __init__(self, parent, workspace, logger):
        self.logger = logger
        
        self.current_spectrum = None
        self.current_region = None
        self.parent = parent
        self.workspace = workspace

        self.init_right_panel()
        self.init_left_panel()

    def init_left_panel(self):
        self.logger.info("Initializing left panel")
        left_panel = QWidget()
        self.left_panel = left_panel
        left_panel_layout = QVBoxLayout()

        self.spectra_tree = SpectraTreeWidget(workspace=self.workspace)
        self.spectra_tree.currentItemChanged.connect(self.set_currents_spectrum)
        self.spectra_tree.currentItemChanged.connect(self.parent.update_viewer)
        self.spectra_tree.currentItemChanged.connect(self.update_region_list)
        self.spectra_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.spectra_tree.setHeaderHidden(True)
        self.spectra_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.spectra_tree.customContextMenuRequested.connect(self.show_context_menu)
        left_panel_layout.addWidget(self.spectra_tree)
        self.update_spectra_tree()

        # Buttons for prediction and postprocessing
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)
        left_panel_layout.addWidget(self.predict_button)

        self.post_process_button = QPushButton("Post process")
        self.post_process_button.clicked.connect(self.post_process)
        left_panel_layout.addWidget(self.post_process_button)

        left_panel.setLayout(left_panel_layout)
    
    def set_currents_spectrum(self, item, column):
        self.current_spectrum = item.data(0, Qt.UserRole)
        self.current_region = None
    
    def set_current_region(self, item):
        if item is not None:
            self.current_region = item.data(Qt.UserRole)
    
    def aggregate_left_panel_items(self):
        selected_items = self.spectra_tree.selectedItems()  # Get all selected items

        if not selected_items:
            return
        
        groups = [item for item in selected_items if not item.parent()]
        spectra = [item for item in selected_items if item.parent()]

        return groups, spectra
    
    def show_context_menu(self, position):
        """Shows a right-click menu with rename & delete options for multiple items."""
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
        new_name, ok = QInputDialog.getText(self.left_panel, "Rename Group", "Enter new group name:")
        if ok and new_name:
            self.workspace.rename_group(item.text(0), new_name)
            item.setText(0, new_name)

    def delete_group(self, item, with_dialog=True):
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
        spectrum = item.data(0, Qt.UserRole)  # Retrieve an object
        if not spectrum:
            return
        new_name, ok = QInputDialog.getText(self.left_panel, "Rename Spectrum", "Enter new name:", text=spectrum.name)
        if ok and new_name:
            spectrum.name = new_name  # Update an object
            item.setText(0, new_name)  # Update UI

    def delete_spectrum(self, item, with_dialog=True):
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
        self.spectra_tree.clear()
        for group, spectra in self.workspace.groups.items():
            group_item = QTreeWidgetItem([group])
            group_item.setFlags(group_item.flags() | Qt.ItemIsDropEnabled)
            for spectrum in spectra:
                spectrum_item = QTreeWidgetItem([spectrum.name])
                spectrum_item.setData(0, Qt.UserRole, spectrum)
                spectrum_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled)
                group_item.addChild(spectrum_item)
            self.spectra_tree.addTopLevelItem(group_item)
    
    def predict(self):
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

        self.workspace.predict(spectra=spectra_list)
        self.parent.toolbar.toggle_labeled_data_action.setChecked(True)
        self.parent.update_viewer()

    def post_process(self):        
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

        self.workspace.post_process(spectra=spectra_list)
        self.parent.toolbar.toggle_lines_action.setChecked(True)
        self.parent.update_viewer()
        self.update_region_list()

    def init_right_panel(self):
        right_panel = QWidget()
        self.right_panel = right_panel
        right_panel_layout = QVBoxLayout()

        # Region parameters
        region_label = QLabel("Optimization tools")
        right_panel_layout.addWidget(region_label)

        # Region list
        self.region_list = QListWidget()
        self.region_list.setFixedHeight(100)
        self.region_list.currentItemChanged.connect(self.set_current_region)
        self.region_list.currentItemChanged.connect(self.load_region_tab)
        self.region_list.currentItemChanged.connect(self.set_cursors)
        right_panel_layout.addWidget(self.region_list)

        refit_layout = QHBoxLayout()
        self.refit_region_button = QPushButton("Optimize")
        self.refit_region_button.clicked.connect(self.refit_region)
        #TODO: update only line tab
        self.refit_region_button.clicked.connect(self.update_lines_settings_tab)
        self.refit_region_button.clicked.connect(self.parent.update_viewer)
        refit_layout.addWidget(self.refit_region_button)

        self.reoptimize_all_box = QCheckBox("Reoptimize all")
        self.reoptimize_all_box.setChecked(False)
        refit_layout.addWidget(self.reoptimize_all_box)

        self.fine_fit = QCheckBox("Fine fit")
        self.fine_fit.setChecked(False)
        refit_layout.addWidget(self.fine_fit)

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
        if self.current_region is not None:
            self.parent.set_cursors(
                    self.current_region, self.current_spectrum
                )

    def update_region_list(self, set_current=True):
        self.logger.info("Updating region list")
        spectrum = self.current_spectrum
        self.region_list.clear()
        if spectrum is not None and len(spectrum.regions) != 0:
            for region in spectrum.regions:
                item = QListWidgetItem(f"Region {spectrum.regions.index(region)}")
                item.setData(Qt.UserRole, region)
                self.region_list.addItem(item)
            if set_current:
                self.region_list.setCurrentRow(0)
            else:
                self.load_region_tab()

    def create_region_tabs(self):
        """
        Creates the tabs for the right panel of the window.
        The tabs are Region settings and Line settings.
        """
        region_info_tab = QWidget()
        tab_layout = QFormLayout()
        region_info_tab.setLayout(tab_layout)
        self.region_tabs.addTab(region_info_tab, "Region settings")

        lines_info_tab = ScrollableWidget()
        add_line_button = QPushButton("Add line")
        add_line_button.clicked.connect(self.add_line)
        lines_info_tab.layout().addWidget(add_line_button)
        self.region_tabs.addTab(lines_info_tab, "Line settings")

    def load_region_tab(self):
        self.clear_tabs()
        if self.current_region is not None:
            self.load_region_settings_tab()
            self.fixed_params_cb = []
            self.load_lines_settings_tab()
    
    def clear_tabs(self):
        self.clear_layout(self.region_tabs.widget(0).layout())
        self.clear_layout(self.region_tabs.widget(1).content_layout)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)  # Remove item
            widget = item.widget()
            if widget:
                widget.deleteLater()  # Delete the widget properly

    def load_region_settings_tab(self):
        self.logger.info("Loading region settings tab")
        region = self.current_region
        tab_layout = self.region_tabs.widget(0).layout()

        start_param = self.create_region_param_input(region, 'start_point') 
        tab_layout.addRow('From (eV):', start_param)

        end_param = self.create_region_param_input(region, 'end_point')
        tab_layout.addRow('To (eV):', end_param)
    
    def update_region_settings_tab(self):
        tab_widget = self.region_tabs.widget(0)
        region = self.current_region
        for param, param_input in zip(('start_point', 'end_point'), tab_widget.findChildren(QLineEdit)):
            param_input.setText(f"{getattr(region, param):.2f}")

    def create_region_param_input(self, region, param):
        current_value = getattr(region, param)
        if not isinstance(current_value, str):
            current_value = f"{current_value:.2f}"
        input_edit = QLineEdit(current_value)
        input_edit.returnPressed.connect(lambda: self.update_region_param(region, param, input_edit.text().strip(), input_edit))
        input_edit.returnPressed.connect(self.parent.update_viewer)
        return input_edit

    def update_region_param(self, region, param, value, edit):
        edit.setText(value)
        self.workspace.change_region_parameter(region, self.current_spectrum, param, value)
    
    #TODO: нумерация линий и апдейт параметров без удаления вкладки после фитинга
    def add_line_to_tab(self, line):
        region = self.current_region
        tab_layout = self.region_tabs.widget(1).content_layout
        # ((parameter1_label, obj parametr), ...)
        editable_params = (
            ('BE', 'loc'),
            ('FWHM', 'fwhm'),
            ('GL ratio', 'gl_ratio'),
            ('Amplitude', 'const')
        )
        noneditable_params = (
            ('Area', 'area'),
            ('Height', 'height')
        )
        line_group = QGroupBox(f"Line {region.lines.index(line)}")
        line_layout = QFormLayout()
        line_group.setLayout(line_layout)
        cb_layout = QHBoxLayout()
    
        cb_label = QLabel("Fix parameters")
        # cb_group = CheckboxGroup(len(editable_params))

        # self.fixed_params_cb.append(cb_group.checkboxes)
        cb_layout.addWidget(cb_label)
        # cb_layout.addWidget(cb_group.main_checkbox)
        cb_layout.setAlignment(Qt.AlignRight)
        line_layout.addRow(cb_layout)
        cb_list = []
        for (param_label, param) in editable_params:
            layout = QHBoxLayout()
            cb = QCheckBox()
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
    
    def load_lines_settings_tab(self):
        region = self.current_region
        for line in region.lines:
            self.add_line_to_tab(line)

    def update_lines_settings_tab(self, save_state=True):
        self.logger.info("Updating lines settings tab")
        tab = self.region_tabs.widget(1)
        region = self.current_region
        for line, line_setting in zip(region.lines, tab.findChildren(QGroupBox)):
            line_setting.setTitle(f"Line {region.lines.index(line)}")
            for param, param_input in zip(('loc', 'fwhm', 'gl_ratio', 'const', 'area', 'height'), line_setting.findChildren(QLineEdit)):
                param_input.setText(f"{getattr(line, param):.2f}")
    
    def remove_line_settings(self, line_idx):
        tab_layout = self.region_tabs.widget(1).content_layout
        widget = tab_layout.itemAt(line_idx).widget()
        tab_layout.removeWidget(widget)
        widget.deleteLater()
        self.fixed_params_cb.pop(line_idx)

    def delete_line(self, line_idx):
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
            input_edit.returnPressed.connect(lambda: self.update_line_param(line, param, input_edit.text().strip(), input_edit))
            input_edit.returnPressed.connect(self.parent.update_viewer)
        return input_edit

    def update_line_param(self, line, param, value, edit):
        edit.setText(value)
        value = float(value)
        self.workspace.change_line_parameter(line, param, value)

    def delete_region(self):
        selected_item = self.region_list.currentItem()
        if selected_item is not None:
            region = selected_item.data(Qt.UserRole)
            self.workspace.delete_region(region, self.current_spectrum)
            self.region_list.takeItem(self.region_list.row(selected_item))
            self.region_list.setCurrentRow(self.region_list.count() - 1)

    def create_new_region(self):
        spectrum = self.current_spectrum
        if spectrum is not None:
            x1 = spectrum.x[0]
            x2 = spectrum.x[-1]
            region = self.workspace.create_new_region(x1, x2, spectrum=spectrum)
            item = QListWidgetItem(f"Region {spectrum.regions.index(region)}")
            item.setData(Qt.UserRole, region)
            self.region_list.addItem(item)
            self.region_list.setCurrentRow(self.region_list.count() - 1)

    #TODO: умное добавление линий
    def add_line(self):
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
        selected_item = self.region_list.currentItem()
        if selected_item is None:
            return
        if self.fine_fit.isChecked():
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

class SpectraTreeWidget(QTreeWidget):
    """Custom QTreeWidget to restrict drag-and-drop behavior."""
    def __init__(self, parent=None, workspace=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QTreeWidget.InternalMove)
        self.viewport().installEventFilter(self)  # Allow drag filtering
        self.wokspace = workspace

    def dropEvent(self, event):
        """Control where items can be dropped."""
        target_item = self.itemAt(event.position().toPoint())
        selected_items = self.selectedItems()

        if not target_item:
            event.ignore()
            return

        # If the target is a spectrum (child node), prevent nesting
        if target_item.parent():
            event.ignore()
            return

        # Move each selected spectrum into the target group
        for item in selected_items:
            if item.parent():  # Only move spectrum, not groups
                old_parent = item.parent()
                old_parent.removeChild(item)  # Remove from old group
                target_item.addChild(item)  # Move to new group
                spectrum = item.data(0, Qt.UserRole)
                idx = self.wokspace.groups[old_parent.text(0)].index(spectrum)
                self.wokspace.move_spectrum(idx, old_parent.text(0), target_item.text(0))

        event.accept()

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

class PlotCanvas(FigureCanvas):
    def __init__(self, workspace, parent):
        self.fig = Figure(figsize=(5, 4))
        self.fig.tight_layout()
        self.ax = self.fig.subplots()
        super().__init__(self.fig)
        self.workspace = workspace
        self._parent = parent

        # No cursors initially
        self.line1 = None
        self.line2 = None
        self.dragging_line = None

        # Connect mouse events
        self.mpl_connect("button_press_event", self.on_click)
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("button_release_event", self.on_release)

    def on_click(self, event):
        if event.inaxes != self.ax or not self.line1 or not self.line2:
            return
        # Detect if click is near one of the lines
        for line in [self.line1, self.line2]:
            if abs(line.get_xdata()[0] - event.xdata) < 0.5:
                self.dragging_line = line
                break

    def on_motion(self, event):
        if self.dragging_line and event.inaxes == self.ax:
            self.dragging_line.set_xdata((event.xdata, ))
            self.draw_idle()  # Refresh plot

    def on_release(self, event):
        if self.dragging_line:
            x1 = self.line1.get_xdata()[0]
            x2 = self.line2.get_xdata()[0]
            x_new = self.dragging_line.get_xdata()[0]

            if x1 == x_new:
                param = 'start_point'
            else:
                param = 'end_point'

            self.dragging_line = None
            if self.spectrum is not None:
                if x1 > x2:
                    x1, x2 = x2, x1
                    self.workspace.change_region_parameter(self.region, self.spectrum, 'start_point', x1)
                    self.workspace.change_region_parameter(self.region, self.spectrum, 'end_point', x2)
                else:
                    self.workspace.change_region_parameter(self.region, self.spectrum, param, x_new)

            self._parent.update_viewer()
            self._parent.sidebars.update_region_settings_tab() #TODO: replace this

    def set_cursors(self, pos1, pos2):
        """Function to set cursors when list item is clicked"""
        # Remove existing lines
        if self.line1:
            self.line1.remove()
        if self.line2:
            self.line2.remove()
        # Create new lines at specified positions
        self.line1 = self.ax.axvline(pos1, color='r', linestyle='--', picker=True)
        self.line2 = self.ax.axvline(pos2, color='r', linestyle='--', picker=True)

        self.draw_idle()  # Update the plot
    
    def get_cursors(self):
        return self.line1.get_xdata()[0], self.line2.get_xdata()[0]

    def plot(self, spectrum, type='lines', add_smoothing=False):
        self.fig.clear()
        self.ax = self.fig.subplots()
        if type == 'lines':
            spectrum.view_lines(self.ax, smoothed=add_smoothing)
            self.ax.set_title(spectrum.name)
        elif type == 'labeled':
            spectrum.view_labeled_data(self.ax)
            self.ax.set_title(f"Labeled Data for {spectrum.name}")
        elif type == 'raw':
            spectrum.view_data(self.ax, smoothed=add_smoothing)
            self.ax.set_title(spectrum.name)
        self.draw()
    
    def load_cursors(self, region, spectrum):
        self.region = region
        self.spectrum = spectrum
        self.set_cursors(region.start_point, region.end_point)


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
