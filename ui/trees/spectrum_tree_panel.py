"""Spectrum tree panel with search, auto-fit, and optimize controls."""

from PySide6.QtCore import QModelIndex, QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget

from ..assets import icon_path
from ..controller import ControllerWrapper
from ..dialogs.optimize_confirm import confirm_and_optimize
from .spectrum_tree import SpectrumTreeModel, SpectrumTreeWidget
from .tree_search import matches_search

_SEARCH_ICON = QIcon(str(icon_path("search.svg")))
_FLASK_ICON = QIcon(str(icon_path("flask.svg")))
_OPTIMIZE_ICON = QIcon(str(icon_path("optimize.svg")))
_BTN_ICON_SIZE = QSize(14, 14)


class SpectrumTreePanel(QWidget):
    """
    Composite widget hosting a search box, Auto fit / Optimize controls, and the spectrum tree.

    The search box filters on each row's ``search_text``. A spectrum row
    includes its own id and the ids of its regions and components, so a peak
    id finds that spectrum. Matching branches are expanded and the first
    match is scrolled into view.
    """

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SpectrumTreePanel")
        self._controller = controller
        self._search_edit = QLineEdit(self)
        self._search_edit.setPlaceholderText("Search spectra, groups, files, IDs…")
        self._search_edit.addAction(_SEARCH_ICON, QLineEdit.ActionPosition.LeadingPosition)
        self._tree = SpectrumTreeWidget(controller, self)
        self._auto_fit_btn = QPushButton("Auto fit", self)
        self._auto_fit_btn.setIcon(_FLASK_ICON)
        self._auto_fit_btn.setIconSize(_BTN_ICON_SIZE)
        self._optimize_btn = QPushButton("Optimize", self)
        self._optimize_btn.setIcon(_OPTIMIZE_ICON)
        self._optimize_btn.setIconSize(_BTN_ICON_SIZE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(4)
        layout.addWidget(self._search_edit)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(4)
        btn_row.addWidget(self._auto_fit_btn)
        btn_row.addWidget(self._optimize_btn)
        layout.addLayout(btn_row)
        layout.addWidget(self._tree)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._search_edit.textChanged.connect(self._on_search_text_changed)
        self._auto_fit_btn.clicked.connect(self._on_auto_fit_clicked)
        self._optimize_btn.clicked.connect(self._on_optimize_clicked)

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
        """Return the underlying spectrum tree model."""
        return self._tree.model

    def refresh(self) -> None:
        """Refresh the tree contents from the controller while keeping the filter."""
        self._tree.refresh()
        self._apply_filter(self._search_edit.text())

    def _on_search_text_changed(self, text: str) -> None:
        """React to search box changes by updating the filter."""
        self._apply_filter(text)

    def _on_auto_fit_clicked(self) -> None:
        """Run segmenter then optimization for all selected spectra."""
        spectrum_ids = self._tree.get_selected_spectrum_ids()
        if not spectrum_ids:
            QMessageBox.information(
                self,
                "No spectrum selected",
                "Select one or more spectra before auto fit.",
            )
            return
        self._controller.run_segmenter(spectrum_ids)
        confirm_and_optimize(self, self._controller, spectrum_ids=spectrum_ids)

    def _on_optimize_clicked(self) -> None:
        """Optimize all regions under each selected spectrum."""
        spectrum_ids = self._tree.get_selected_spectrum_ids()
        if not spectrum_ids:
            QMessageBox.information(
                self,
                "No spectrum selected",
                "Select one or more spectra before optimizing.",
            )
            return
        confirm_and_optimize(self, self._controller, spectrum_ids=spectrum_ids)

    def _apply_filter(self, text: str) -> None:
        """
        Hide rows that do not match ``text``.

        A row stays visible when ``text`` is in its ``search_text`` or label,
        or in any descendant. Ancestors of a match are expanded, and the view
        scrolls to the first match.

        Parameters
        ----------
        text : str
            Search text entered by the user.
        """
        query = text.strip().lower()
        self._scroll_target = QModelIndex()

        if not query:
            self._show_all(QModelIndex())
            self._tree.expandAll()
            return

        self._visit(QModelIndex(), False, query)
        if self._scroll_target.isValid():
            self._tree.scrollTo(self._scroll_target)

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
            matched = item is not None and matches_search(query, item.search_text, item.label)

            child_has_match = self._visit(index, ancestor_visible or matched, query)
            has_match_here = matched or child_has_match
            visible = has_match_here or ancestor_visible
            self._tree.setRowHidden(row, parent_index, not visible)

            if has_match_here:
                any_visible = True
                self._expand_ancestors(index)
                if matched and not self._scroll_target.isValid():
                    self._scroll_target = index
        return any_visible

    def _expand_ancestors(self, index: QModelIndex) -> None:
        """Expand every parent of ``index`` so a match is reachable."""
        parent = index.parent()
        while parent.isValid():
            self._tree.expand(parent)
            parent = parent.parent()
