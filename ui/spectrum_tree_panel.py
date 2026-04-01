from typing import Any

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import QLineEdit, QVBoxLayout, QWidget

from .controller import ControllerWrapper
from .spectrum_tree import SpectrumTreeModel, SpectrumTreeWidget


class SpectrumTreePanel(QWidget):
    """
    Composite widget hosting a search box and the spectrum tree.

    The search box filters and highlights spectra, groups, and files by
    matching the entered text against their labels. Matching branches are
    expanded automatically.
    """

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._search_edit = QLineEdit(self)
        self._search_edit.setPlaceholderText("Search spectra, groups, files…")
        self._tree = SpectrumTreeWidget(controller, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._search_edit)
        layout.addWidget(self._tree)

        self._search_edit.textChanged.connect(self._on_search_text_changed)

    @property
    def tree(self) -> SpectrumTreeWidget:
        """
        Return the underlying spectrum tree widget.

        Returns
        -------
        SpectrumTreeWidget
            Tree view showing the spectra hierarchy.
        """
        return self._tree

    @property
    def model(self) -> SpectrumTreeModel:
        """
        Return the underlying spectrum tree model.
        """
        return self._tree.model

    def refresh(self) -> None:
        """
        Refresh the tree contents from the controller while keeping the filter.
        """
        self._tree.refresh()
        self._apply_filter(self._search_edit.text())

    def _on_search_text_changed(self, text: str) -> None:
        """React to search box changes by updating the filter."""
        self._apply_filter(text)

    def _apply_filter(self, text: str) -> None:
        """
        Apply a simple filter to the tree by hiding non-matching items.

        Parameters
        ----------
        text : str
            Search text entered by the user.
        Items whose own label or any descendant label contains the query
        remain visible; all other branches are hidden.
        """
        query = text.strip().lower()

        if not query:
            self._show_all(QModelIndex())
            self._tree.expandAll()
            return

        self._visit(QModelIndex(), False, query)

    def _show_all(self, parent_index: QModelIndex) -> None:
        row_count = self.model.rowCount(parent_index)
        for row in range(row_count):
            index = self.model.index(row, 0, parent_index)
            self._tree.setRowHidden(row, parent_index, False)
            self._show_all(index)

    def _visit(self, parent_index: QModelIndex, ancestor_visible: bool, query: str) -> bool:
        """
        Return True if any child (or the item itself) under parent_index matches.

        If a parent row is visible, all of its children remain visible regardless
        of whether they individually match the query.
        """
        any_visible = False
        row_count = self.model.rowCount(parent_index)
        for row in range(row_count):
            index = self.model.index(row, 0, parent_index)
            item = self.model.item_from_index(index)

            label = item.label.lower() if item is not None else ""
            matched = query in label

            child_has_match = self._visit(index, ancestor_visible or matched, query)
            has_match_here = matched or child_has_match
            visible = has_match_here or ancestor_visible
            self._tree.setRowHidden(row, parent_index, not visible)

            if has_match_here:
                any_visible = True
                if matched:
                    self._tree.expand(index)
        return any_visible
