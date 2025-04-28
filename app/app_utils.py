from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import *

class TreeWithSearch(QTreeWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search...")
        self.search_box.textChanged.connect(self.filter_tree)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def filter_tree(self, text):
        text = text.lower()
        for i in range(self.topLevelItemCount()):
            file_item = self.topLevelItem(i)
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
