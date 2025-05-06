from itertools import chain

from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from PySide6.QtWidgets import *

from app.analysis_window import AnalysisWindow
from app.app_utils import TreeWithSearch, ProgressBarWindow, ScrollableWidget, FittingWindow

class Sidebars():
    def __init__(self, parent, workspace, logger):
        self.logger = logger

        self.parent = parent
        self.workspace = workspace
        
        self.current_spectrum = None
        self.current_region = None
        self.analysis_window = None

        self.copied_spectrum = None

        self.init_right_panel()
        self.init_left_panel()

    def init_left_panel(self):
        self.logger.debug("Initializing left panel")
        left_panel = QWidget()
        self.left_panel = left_panel
        left_panel_layout = QVBoxLayout()

        self.spectra_tree = TreeWithSearch()
        left_panel_layout.addWidget(self.spectra_tree.search_box)
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

        # Automatic analysis and options
        layout_1 = QHBoxLayout()
        layout_2 = QHBoxLayout()
        left_panel_layout.addLayout(layout_1)
        left_panel_layout.addLayout(layout_2)

        automatic_analysis_button = QPushButton("Automatic analysis")
        # automatic_analysis_button.setFixedSize(150, 30)
        automatic_analysis_button.clicked.connect(self.automatic_analysis)
        layout_1.addWidget(automatic_analysis_button)

        change_prediction_threshold_button = QPushButton("Set threshold")
        change_prediction_threshold_button.clicked.connect(self.change_prediction_threshold)
        # change_prediction_threshold_button.setFixedSize(150, 30)
        layout_2.addWidget(change_prediction_threshold_button)

        self.force_analysis_box = QCheckBox("Force analysis")
        self.force_analysis_box.setChecked(False)
        layout_2.addWidget(self.force_analysis_box)

        self.skip_survey_box = QCheckBox("Skip survey")
        self.skip_survey_box.setChecked(True)
        layout_2.addWidget(self.skip_survey_box)

        # Trend analysis
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

    def change_prediction_threshold(self):
        self.logger.debug("Changing prediction threshold")
        value = self.workspace.pred_threshold
        new_threshold, ok = QInputDialog.getDouble(self.left_panel, "Change Prediction Threshold", "Enter new prediction threshold:", value, 0, 1, 2, step=0.01)
        if ok:
            self.workspace.set_prediction_threshold(new_threshold)
        self.parent.update_viewer()

    def aggregate_left_panel_items(self):
        self.logger.debug("Aggregating left panel items")
        selected_items = self.spectra_tree.selectedItems()  # Get all selected items

        files = [item.text(0) for item in selected_items if not item.parent()]
        groups = [item.text(0) for item in selected_items if item.parent() and item.childCount() > 0]
        spectra = [item.data(0, Qt.UserRole) for item in selected_items if item.parent() and item.childCount() == 0]

        return files, groups, spectra
    
    def show_context_menu(self, position):
        """Shows a right-click menu with rename, delete, move, copy and paste options for multiple items."""
        self.logger.debug("Showing context menu")
        files, groups, spectra = self.aggregate_left_panel_items()

        if len(files) + len(groups) + len(spectra) == 0:
            return  # No item was clicked

        left_panel = self.left_panel
        menu = QMenu(left_panel)

        if len(groups) > 1 and not spectra and not files: # If multiple groups are selected, show "Merge groups"
            merge_action = QAction("Merge groups", left_panel)
            merge_action.triggered.connect(lambda: self.merge_selected(groups))
            menu.addAction(merge_action)
        elif len(spectra) >= 1 and not groups and not files: # If at least one spectrum is selected, show "Move spectra" and "Paste"
            paste_action = QAction("Paste spectrum", left_panel)
            paste_action.triggered.connect(lambda: self.paste_spectrum(spectra))
            menu.addAction(paste_action)
            if not self.copied_spectrum:
                paste_action.setEnabled(False)
            
            move_action = QAction("Move spectra", left_panel)
            move_action.triggered.connect(lambda: self.move_selected_spectra(spectra))
            menu.addAction(move_action)

            check_survey = QAction("Check survey", left_panel, checkable=True)
            if len([s for s in spectra if s.is_survey]) == len(spectra):
                check_survey.setChecked(True)
            else:
                check_survey.setChecked(False)
            check_survey.changed.connect(lambda: self.check_survey(spectra, check=check_survey.isChecked()))
            menu.addAction(check_survey)

        if len(spectra) == 1 and not groups and not files: # If one spectrum is selected, show "Copy" and "Rename"
            copy_action = QAction("Copy spectrum", left_panel)
            copy_action.triggered.connect(lambda: self.copy_spectrum(spectra[0]))
            menu.addAction(copy_action)
            rename_action = QAction("Rename", left_panel)
            rename_action.triggered.connect(lambda: self.rename_spectra(spectra[0]))
            menu.addAction(rename_action)
        elif len(groups) == 1 and not files and not spectra: # If one group is selected, show "Rename"
            rename_action = QAction("Rename", left_panel)
            rename_action.triggered.connect(lambda: self.rename_group(groups[0]))
            menu.addAction(rename_action)

        if len(files) + len(groups) + len(spectra) >= 1: # If at least one item is selected, show "Delete"
            delete_action = QAction("Delete", left_panel)
            delete_action.triggered.connect(lambda: self.delete_selected_items(spectra, files, groups))
            menu.addAction(delete_action)

        menu.exec(self.spectra_tree.viewport().mapToGlobal(position))

    def merge_selected(self, group_names):
        """Merges multiple selected groups or files into one."""
        self.logger.debug("Merging selected groups")
        new_group_name, ok = QInputDialog.getText(self.left_panel, "Merge groups", "Enter new group name:", text=f"{group_names[0]}")
        if not ok or not new_group_name:
            return  # User canceled
        self.workspace.merge_groups(new_group_name, group_names)
        self.update_spectra_tree()
    
    def paste_spectrum(self, spectra):
        """Paste regions from copied spectrum to selected spectra"""
        self.logger.debug("Pasting spectrum")
        self.workspace.paste_spectra(self.copied_spectrum, spectra)
        self.load_region_tab()
        self.copied_spectrum = None

    def move_selected_spectra(self, spectra):
        """Move selected spectra to a new group."""
        self.logger.debug("Moving selected spectra")
        new_group_name, ok = QInputDialog.getText(self.left_panel, "Move spectra", "Enter new group name:")
        if ok and new_group_name:
            self.workspace.move_spectra(spectra, new_group_name)
            self.update_spectra_tree()

    def check_survey(self, spectra, check=True):
        self.logger.debug("Checking survey")
        self.workspace.check_survey(spectra, check)
    
    def copy_spectrum(self, spectrum):
        self.logger.debug("Copying selected spectrum")
        self.copied_spectrum = spectrum
    
    def rename_spectra(self, spectrum):
        self.logger.debug("Renaming spectrum")
        new_name, ok = QInputDialog.getText(self.left_panel, "Rename spectrum", "Enter new name:", text=spectrum.name)
        if ok and new_name:
            self.workspace.rename_spectrum(spectrum, new_name)
            self.update_spectra_tree()

    def rename_group(self, group):
        self.logger.debug("Renaming group")
        new_name, ok = QInputDialog.getText(self.left_panel, "Rename group", "Enter new group name:")
        if ok and new_name:
            self.workspace.rename_group(group, new_name)
            self.update_spectra_tree()

    def delete_selected_items(self, spectra, files, groups):
        self.logger.debug("Deleting selected items")
        l = len(spectra) + len(files) + len(groups)
        confirm = QMessageBox.question(
            self.left_panel, "Delete selected items",
            f"Are you sure you want to delete {l} selected items?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.workspace.delete_spectra(spectra=spectra, files=files, groups=groups)
            self.update_spectra_tree()

    def update_spectra_tree(self):
        self.logger.debug("Updating spectra tree")
        self.spectra_tree.clear()
        tree = {}
        # construct tree
        for s in self.workspace.spectra:
            file = s.file
            group = s.group
            spectrum_item = QTreeWidgetItem([s.name])
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
    
    def automatic_analysis(self):
        self.logger.debug("Automatic analysis")
        skip_survey = self.skip_survey_box.isChecked()
        files, groups, spectra = self.aggregate_left_panel_items()
        spectra = self.workspace.aggregate_unique_spectra(spectra, files, groups, skip_survey)
        if spectra is None or len(spectra) == 0:
            spectra = self.workspace.aggregate_spectra()
        
        # predict
        not_predicted = [s for s in spectra if not s.is_predicted]
        self.workspace.predict(spectra=not_predicted)

        # post process with progress bar
        if self.force_analysis_box.isChecked(): # analyze all selected
            # drop previous regions
            for s in spectra:
                s.regions = []
                s.is_analyzed = False
        not_analyzed = [s for s in spectra if not s.is_analyzed and s.is_predicted]
        progress_window = ProgressBarWindow(self.workspace.post_process, len(not_analyzed), not_analyzed)
        progress_window.exec()

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
        self.create_region_button = QPushButton("Create region")
        self.create_region_button.clicked.connect(self.create_new_region)
        self.create_region_button.clicked.connect(self.parent.update_viewer)
        self.create_region_button.clicked.connect(self.update_region_list)
        create_delete_layout.addWidget(self.create_region_button)

        delete_button = QPushButton("Delete region")
        delete_button.clicked.connect(self.delete_region)
        delete_button.clicked.connect(self.parent.update_viewer)
        delete_button.clicked.connect(self.update_region_list)
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
            if self.region_list.count() > 0:
                item_idx = self.region_list.count() - 1
                self.region_list.setCurrentRow(item_idx)
                item = self.region_list.item(item_idx)
                self.set_current_region(item)
            else:
                # If all regions are deleted the spectrum is no longer analyzed
                self.current_spectrum.is_analyzed = False
                self.current_region = None

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
            self.set_current_region(item)

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
            fw = FittingWindow(self.workspace.refit, region, tol=0.15, full_refit=True, fit_alg=fit_alg)
            fw.exec()
        else:
            fixed_params = self.parse_fixed_params()
            fw = FittingWindow(self.workspace.refit, region, fixed_params=fixed_params, tol=0.2, fit_alg=fit_alg, loc_tol=1)
            fw.exec()

    def parse_fixed_params(self):
        return [i for i, box in enumerate(chain(*self.fixed_params_cb)) if box.isChecked()]
