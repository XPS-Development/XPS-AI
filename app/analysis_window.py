import numpy as np
import pyqtgraph as pg

from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import *

from app.app_utils import TreeWithSearch


class AnalysisWindow(QDialog):
    def __init__(self, workspace):
        super().__init__()
        self.setWindowTitle("Analysis Window")
        self.setGeometry(100, 100, 700, 400)
        self.workspace = workspace

        self.attrs = ('loc', 'const', 'fwhm', 'gl_ratio')
        self.header = ('Position', 'Area', 'FWHM', 'GL')
        self.fmt = ('{:.2f}', '{:.2f}', '{:.2f}', '{:.2f}')

        layout = QVBoxLayout()

        # Initial tree widget list
        self.tree_widget = TreeWithSearch()
        self.tree_widget.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tree_widget.setHeaderHidden(True)
        self.populate_tree()
        tree_layout = QVBoxLayout()
        tree_layout.addWidget(QLabel("Available objects:"))
        tree_layout.addWidget(self.tree_widget.search_box)
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
        list_layout.addWidget(QLabel("Selected objects:"))
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

        btn_proceed = QPushButton("Plot")
        btn_proceed.clicked.connect(self.view)
        btn_layout_bottom.addWidget(btn_proceed)

        # Add widgets to layout
        layout.addLayout(lists_layout)
        layout.addLayout(btn_layout_bottom)

        self.setLayout(layout)
        self.tree_widget.setFocus() # prevent editing of search box

    def populate_tree(self):
        # self.logger.debug("Updating spectra tree")
        self.tree_widget.clear()
        tree = {}
        # construct tree
        for s in self.workspace.spectra:
            if not s.is_analyzed:
                continue
            file = s.file
            group = s.group
            spectrum_item = QTreeWidgetItem([s.name])
            if file is None:
                file = "Unsorted"
            if group is None:
                group = "Unsorted"
            if file not in tree:
                file_item = QTreeWidgetItem([file])
                tree[file] = (file_item, {})
                self.tree_widget.addTopLevelItem(file_item)
            if group not in tree[file][1]:
                group_item = QTreeWidgetItem([group])
                tree[file][1][group] = group_item
                tree[file][0].addChild(group_item)
            tree[file][1][group].addChild(spectrum_item)
            i = 0
            for reg_n in s.regions:
                for p in reg_n.lines:
                    peak_name = f"Peak {i} at {p.loc:.1f}"
                    peak_item = QTreeWidgetItem([peak_name])
                    peak_item.setData(0, Qt.UserRole, p)
                    spectrum_item.addChild(peak_item)
                    i += 1
        self.tree_widget.expandAll()
    
    def highlight_added_items(self):
        for i in range(self.tree_widget.topLevelItemCount()):
            file_item = self.tree_widget.topLevelItem(i)
            all_childs = self.traverse(file_item)
            for item in all_childs:
                if item.data(0, Qt.UserRole) in self.selected_objects:
                    item.setBackground(0, QtGui.QColor(0, 255, 0, 30))
    
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
            if not (item.data(0, Qt.UserRole) in self.selected_objects or item.isHidden()):
                self.selected_objects.append(item.data(0, Qt.UserRole))
        self.highlight_added_items()
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
        self.highlight_added_items()
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
        self.plot_dialog.setWindowTitle("Trend plot")
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
        color = self.palette().color(QtGui.QPalette.Text)
        plot_widget.plot(
            x, y, pen={'width': 2, 'color': color}, 
            symbol='o', symbolPen={'color': color}, 
            symbolBrush=color
        )

        self.plot_dialog.show()
