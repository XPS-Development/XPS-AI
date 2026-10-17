"""Choose spectra from several branches and arrange an explicit series order."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.trees.grouping import iter_spectrum_groups

if TYPE_CHECKING:
    from ui.controller import ControllerWrapper


class SeriesDialog(QDialog):
    """Select and reorder spectra without deriving order from subsequent table sorting."""

    def __init__(
        self,
        controller: ControllerWrapper,
        parent: QWidget | None = None,
        *,
        name: str = "New series",
        spectrum_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spectrum series")
        self.setObjectName("PostProcessingDialog")
        self.resize(780, 520)
        self.name_edit = QLineEdit(name)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search files, groups, spectra or IDs…")
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Decomposed spectra")
        self.ordered = QListWidget()
        self.ordered.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._leaves: dict[str, QTreeWidgetItem] = {}
        self._labels: dict[str, str] = {}
        excluded = {
            sid
            for sid in controller.query.get_all_spectra_ids()
            if controller.query.get_spectrum_structure_status(sid) != "peaks"
        }
        for file in iter_spectrum_groups(controller.query, skip_ids=excluded):
            file_item = QTreeWidgetItem(self.tree, [file.label])
            self._checkable(file_item)
            for group in file.groups:
                group_item = QTreeWidgetItem(file_item, [group.label])
                self._checkable(group_item)
                for spectrum in group.spectra:
                    item = QTreeWidgetItem(group_item, [spectrum.label])
                    self._checkable(item)
                    item.setData(0, Qt.ItemDataRole.UserRole, spectrum.spectrum_id)
                    item.setToolTip(0, spectrum.spectrum_id)
                    self._leaves[spectrum.spectrum_id] = item
                    self._labels[spectrum.spectrum_id] = item.text(0)
        for spectrum_id in dict.fromkeys(spectrum_ids):
            self._append(spectrum_id)
            if spectrum_id in self._leaves:
                self._leaves[spectrum_id].setCheckState(0, Qt.CheckState.Checked)
        self.tree.expandAll()
        self._renumber()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.addWidget(self.search)
        left_layout.addWidget(self.tree, 1)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.addWidget(QLabel("Series steps — drag to reorder"))
        right_layout.addWidget(self.ordered, 1)
        row = QHBoxLayout()
        for label, callback in (
            ("Up", lambda: self._move(-1)),
            ("Down", lambda: self._move(1)),
            ("Remove", self._remove),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        right_layout.addLayout(row)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("Series name", self.name_edit)
        layout.addLayout(form)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.buttons)
        self.tree.itemChanged.connect(self._item_changed)
        self._update_branches()
        self.search.textChanged.connect(self._filter)
        self.name_edit.textChanged.connect(self._validate)
        self.ordered.model().rowsMoved.connect(self._renumber)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self._validate()

    @staticmethod
    def _checkable(item: QTreeWidgetItem) -> None:
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, Qt.CheckState.Unchecked)

    def spectrum_ids(self) -> tuple[str, ...]:
        """Return the explicit user order, with each spectrum represented once."""
        return tuple(
            self.ordered.item(index).data(Qt.ItemDataRole.UserRole)
            for index in range(self.ordered.count())
        )

    def _append(self, spectrum_id: str) -> None:
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, spectrum_id)
        item.setToolTip(spectrum_id)
        self.ordered.addItem(item)

    def _visible_leaves(self, item: QTreeWidgetItem) -> list[QTreeWidgetItem]:
        if item.isHidden():
            return []
        if not item.childCount():
            return [item]
        return [
            leaf for i in range(item.childCount()) for leaf in self._visible_leaves(item.child(i))
        ]

    def _item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        with QSignalBlocker(self.tree):
            if item.childCount():
                state = item.checkState(0)
                for leaf in self._visible_leaves(item):
                    leaf.setCheckState(0, state)
        self._update_branches()
        self._sync_selection()

    def _update_branches(self) -> None:
        def update(item: QTreeWidgetItem) -> None:
            for i in range(item.childCount()):
                update(item.child(i))
            if item.childCount():
                leaves = self._visible_leaves(item)
                count = sum(leaf.checkState(0) == Qt.CheckState.Checked for leaf in leaves)
                state = (
                    Qt.CheckState.Unchecked
                    if not count
                    else Qt.CheckState.Checked
                    if count == len(leaves)
                    else Qt.CheckState.PartiallyChecked
                )
                item.setCheckState(0, state)

        with QSignalBlocker(self.tree):
            for i in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(i)
                if item is not None:
                    update(item)

    def _sync_selection(self) -> None:
        checked = {
            sid for sid, item in self._leaves.items() if item.checkState(0) == Qt.CheckState.Checked
        }
        for index in reversed(range(self.ordered.count())):
            sid = self.ordered.item(index).data(Qt.ItemDataRole.UserRole)
            if sid in self._leaves and sid not in checked:
                self.ordered.takeItem(index)
        selected = set(self.spectrum_ids())
        for sid in self._leaves:
            if sid in checked and sid not in selected:
                self._append(sid)
        self._renumber()

    def _renumber(self) -> None:
        for index, sid in enumerate(self.spectrum_ids()):
            self.ordered.item(index).setText(
                f"{index + 1}. {self._labels.get(sid, f'Unavailable [{sid[:5]}]')}"
            )
        if hasattr(self, "buttons"):
            self._validate()

    def _move(self, offset: int) -> None:
        index = self.ordered.currentRow()
        target = index + offset
        if index >= 0 and 0 <= target < self.ordered.count():
            self.ordered.insertItem(target, self.ordered.takeItem(index))
            self.ordered.setCurrentRow(target)
            self._renumber()

    def _remove(self) -> None:
        index = self.ordered.currentRow()
        if index < 0:
            return
        sid = self.ordered.item(index).data(Qt.ItemDataRole.UserRole)
        if sid in self._leaves:
            self._leaves[sid].setCheckState(0, Qt.CheckState.Unchecked)
        else:
            self.ordered.takeItem(index)
            self._renumber()

    def _validate(self) -> None:
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            bool(self.name_edit.text().strip()) and self.ordered.count() > 0
        )

    def _filter(self, text: str) -> None:
        needle = text.strip().casefold()

        def show(item: QTreeWidgetItem, inherited: bool = False) -> bool:
            own = (
                inherited
                or needle
                in (
                    item.text(0) + " " + str(item.data(0, Qt.ItemDataRole.UserRole) or "")
                ).casefold()
            )
            children = [show(item.child(index), own) for index in range(item.childCount())]
            visible = own or any(children)
            item.setHidden(not visible)
            return visible

        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item is not None:
                show(item)
        self._update_branches()
