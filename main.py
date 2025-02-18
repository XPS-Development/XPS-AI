import sys

import torch

from PySide6.QtWidgets import \
    QApplication, QWidget, QVBoxLayout, QFileDialog, QMainWindow, \
    QListWidget, QListWidgetItem, QToolBar, QSplitter, \
    QHBoxLayout, QPushButton, QMessageBox, QInputDialog, QMenu

from PySide6.QtGui import QAction, QPalette, QColor
from PySide6.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from model.models.model_deeper import XPSModel
from tools import Workspace

#TODO: правая панель с настройками спектров - регионы, бекграунд, нормализация, линии и фитинг
#TODO: мердж групп и отображение из спектров
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectral Data Processing App")
        self.setGeometry(100, 100, 800, 600)

        m = self.load_model()
        self.workspace = Workspace(model=m)  # Initialize your workspace

        self.initUI()

    def load_model(self):
        m = XPSModel()
        m.load_state_dict(
            torch.load('model.pt', map_location=torch.device('cpu'), weights_only=True)
        )
        m.eval()
        return m

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Toolbar for actions
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)

        # Load Spectra Action
        load_action = QAction("Load Spectra", self)
        load_action.triggered.connect(self.load_spectra)
        self.toolbar.addAction(load_action)

        # Splitter for sidebar and main content
        splitter = QSplitter()

        # Sidebar layout
        sidebar_layout = QVBoxLayout()

        # List for groups
        self.group_list = QListWidget()
        self.group_list.itemClicked.connect(self.update_spectra_list)
        self.group_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.group_list.setFixedHeight(150)
        self.group_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.group_list.customContextMenuRequested.connect(self.show_group_context_menu)
        sidebar_layout.addWidget(self.group_list)

        # Panel for currently loaded spectra
        self.spectra_list = QListWidget()
        self.spectra_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.spectra_list.itemClicked.connect(self.view_spectrum)
        self.spectra_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.spectra_list.customContextMenuRequested.connect(self.show_spectrum_context_menu)
        sidebar_layout.addWidget(self.spectra_list)

        # Buttons for group actions
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)
        sidebar_layout.addWidget(self.predict_button)

        self.post_process_button = QPushButton("Post process")
        self.post_process_button.clicked.connect(self.post_process)
        sidebar_layout.addWidget(self.post_process_button)

        # Buttons for spectrum actions
        self.show_labeled_data_button = QPushButton("Show Labeled Data")
        self.show_labeled_data_button.clicked.connect(self.show_labeled_data)
        sidebar_layout.addWidget(self.show_labeled_data_button)

        self.show_lines_button = QPushButton("Show Lines")
        self.show_lines_button.clicked.connect(self.show_lines)
        sidebar_layout.addWidget(self.show_lines_button)

        # Sidebar widget
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar_layout)
        splitter.addWidget(sidebar_widget)

        # Main content layout
        main_content_layout = QVBoxLayout()

        # Figure for plotting spectra
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        main_content_layout.addWidget(self.canvas)

        # Main content widget
        main_content_widget = QWidget()
        main_content_widget.setLayout(main_content_layout)
        splitter.addWidget(main_content_widget)

        # Set the splitter to the central widget
        central_layout = QHBoxLayout()
        central_layout.addWidget(splitter)
        central_widget.setLayout(central_layout)

    def load_spectra(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Load Spectra", "", "All Files (*);;Vamas Files (*.vms);;Text Files (*.txt);;SPECS Files (*.xml)")
        if files:
            self.workspace.load_files(*files)
            self.update_group_list()

    def update_group_list(self):
        self.group_list.clear()
        self.group_list.addItems(self.workspace.groups.keys())
        if self.group_list.count() > 0:
            self.group_list.setCurrentRow(0)

    def update_spectra_list(self):
        self.spectra_list.clear()
        selected_items = self.group_list.selectedItems()
        if selected_items:
            group_name = selected_items[0].text()
            if group_name in self.workspace.groups:
                for spectrum in self.workspace.groups[group_name]:
                    item = QListWidgetItem(f"{spectrum.name}")
                    item.setData(Qt.UserRole, spectrum)
                    self.spectra_list.addItem(item)

    def view_spectrum(self, item):
        spectrum = item.data(Qt.UserRole)
        self.plot_spectrum(spectrum)

    def plot_spectrum(self, spectrum):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        spectrum.view_data(ax)
        ax.set_title(spectrum.name)
        self.canvas.draw()

    def predict(self):
        selected_groups = self.group_list.selectedItems()
        if len(selected_groups) > 1:
            self.workspace.predict([item.text() for item in selected_groups])
            print(f"Prediction completed for {len(selected_groups)} groups.")
        elif len(selected_groups) == 1:
            selected_spectra = self.spectra_list.selectedItems()
            if selected_spectra:
                self.workspace.predict(spectra=[item.data(Qt.UserRole) for item in selected_spectra])
                print(f"Prediction completed for {len(selected_spectra)} spectra.")
            else:
                self.workspace.predict([selected_groups[0].text()])
                print(f"Prediction completed for {selected_groups[0].text()}.")

    def post_process(self):
        selected_groups = self.group_list.selectedItems()
        if len(selected_groups) > 1:
            self.workspace.post_process([item.text() for item in selected_groups])
            print(f"Post processing completed for {len(selected_groups)} groups.")
        elif len(selected_groups) == 1:
            selected_spectra = self.spectra_list.selectedItems()
            if selected_spectra:
                self.workspace.post_process(spectra=[item.data(Qt.UserRole) for item in selected_spectra])
                print(f"Post processing completed for {len(selected_spectra)} spectra.")
            else:
                self.workspace.post_process([selected_groups[0].text()])
                print(f"Post processing completed for {selected_groups[0].text()}.")

    def rename_group(self):
        selected_items = self.group_list.selectedItems()
        if selected_items:
            current_group = selected_items[0].text()
            new_name, ok = QInputDialog.getText(self, "Rename Group", "Enter new group name:")
            if ok and new_name:
                self.workspace.rename_group(current_group, new_name)
                self.update_group_list()

    def delete_group(self):
        selected_items = self.group_list.selectedItems()
        if selected_items:
            current_group = selected_items[0].text()
            reply = QMessageBox.question(self, "Delete Group", f"Are you sure you want to delete the '{current_group}'?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.workspace.delete_group(current_group)
                self.update_group_list()

    def merge_groups(self):
        selected_items = self.group_list.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Merge Groups", "Please select at least two groups to merge.")
            return

        new_group_name, ok = QInputDialog.getText(self, "Merge Groups", "Enter new group name:")
        if ok and new_group_name:
            merged_group = []
            for item in selected_items:
                spectrum = item.data(Qt.UserRole)
                merged_group.append(spectrum)

            self.workspace.create_group(new_group_name)
            self.workspace.groups[new_group_name].extend(merged_group)

            for item in selected_items:
                group_name = self.group_list.currentItem().text()
                spectrum = item.data(Qt.UserRole)
                self.workspace.groups[group_name].remove(spectrum)

            self.update_group_list()
            self.update_spectra_list()

    def show_group_context_menu(self, position):
        menu = QMenu()
        rename_action = menu.addAction("Rename Group")
        delete_action = menu.addAction("Delete Group")
        merge_action = menu.addAction("Merge Groups")

        action = menu.exec(self.group_list.viewport().mapToGlobal(position))

        if action == rename_action:
            self.rename_group()
        elif action == delete_action:
            self.delete_group()
        elif action == merge_action:
            self.merge_groups()

    def show_spectrum_context_menu(self, position):
        menu = QMenu()
        merge_action = menu.addAction("Merge Spectra")

        action = menu.exec(self.spectra_list.viewport().mapToGlobal(position))

        if action == merge_action:
            self.merge_groups()

    def show_labeled_data(self):
        selected_items = self.spectra_list.selectedItems()
        if selected_items:
            spectrum = selected_items[0].data(Qt.UserRole)
            self.plot_labeled_data(spectrum)

    def show_lines(self):
        selected_items = self.spectra_list.selectedItems()
        if selected_items:
            spectrum = selected_items[0].data(Qt.UserRole)
            self.plot_lines(spectrum)

    def plot_labeled_data(self, spectrum):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        spectrum.view_labeled_data(ax)
        ax.set_title(f"Labeled Data for {spectrum.name}")
        self.canvas.draw()

    def plot_lines(self, spectrum):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        spectrum.view_lines(ax)
        ax.set_title(f"Lines for {spectrum.name}")
        self.canvas.draw()


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
