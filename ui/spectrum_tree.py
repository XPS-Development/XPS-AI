from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from PySide6.QtCore import QAbstractItemModel, QModelIndex, QPoint, Qt
from PySide6.QtWidgets import QApplication, QInputDialog, QMenu, QTreeView, QWidget

from .controller import ControllerWrapper


@dataclass
class SpectrumTreeItem:
    """
    Node in the spectrum tree hierarchy.

    Each item represents either a file, a group within a file, or a single
    spectrum leaf node. Only spectrum nodes carry a ``spectrum_id``; parent
    nodes are used purely for grouping.

    Parameters
    ----------
    label : str
        Text shown in the tree view.
    kind : str
        Item type identifier (e.g. ``\"file\"``, ``\"group\"``, ``\"spectrum\"``).
    parent : SpectrumTreeItem or None, optional
        Parent item in the hierarchy.
    spectrum_id : str or None, optional
        Identifier of the spectrum associated with this item (for leaf nodes).
    """

    _label: str
    label: str
    kind: str
    parent: Optional["SpectrumTreeItem"] = None
    spectrum_id: Optional[str] = None
    _row: int = 0
    children: list["SpectrumTreeItem"] = field(default_factory=list)

    def child(self, row: int) -> Optional["SpectrumTreeItem"]:
        """Return child at the given row index."""
        if 0 <= row < len(self.children):
            return self.children[row]
        return None

    def row(self) -> int:
        return self._row

    def append_child(self, item: "SpectrumTreeItem") -> None:
        item._row = len(self.children)
        self.children.append(item)


class SpectrumTreeModel(QAbstractItemModel):
    """
    Tree model exposing File → Group → Spectrum hierarchy.

    The model reads all spectra and their metadata from a
    :class:`ControllerWrapper` instance. It groups spectra first by
    ``metadata.file`` and then by ``metadata.group``. Spectra without metadata
    are placed under generic \"No file\" / \"No group\" buckets.
    """

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._root_item = SpectrumTreeItem(_label="Root", label="Root", kind="root")
        self.refresh()

    # ------------------------------------------------------------------
    # Required model API
    # ------------------------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if not parent.isValid():
            item = self._root_item
        else:
            item = parent.internalPointer()
        if not isinstance(item, SpectrumTreeItem):
            return 0
        return len(item.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        del parent
        return 1

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:  # noqa: N802
        if column != 0 or row < 0:
            return QModelIndex()

        if not parent.isValid():
            parent_item = self._root_item
        else:
            parent_item = parent.internalPointer()

        if not isinstance(parent_item, SpectrumTreeItem):
            return QModelIndex()

        child_item = parent_item.child(row)
        if child_item is None:
            return QModelIndex()
        return self.createIndex(row, column, child_item)

    def parent(self, index: QModelIndex) -> QModelIndex:  # noqa: N802
        if not index.isValid():
            return QModelIndex()

        item = index.internalPointer()
        if not isinstance(item, SpectrumTreeItem):
            return QModelIndex()

        parent_item = item.parent
        if parent_item is None or parent_item is self._root_item:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if not index.isValid():
            return None

        item = index.internalPointer()
        if not isinstance(item, SpectrumTreeItem):
            return None

        if role in (Qt.DisplayRole, Qt.EditRole):
            return item.label

        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:  # noqa: N802
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Rebuild tree from controller data."""

        self.beginResetModel()
        self._root_item.children.clear()

        params = self._controller.get_app_parameters()

        grouped = defaultdict(lambda: defaultdict(list))

        for spectrum_id in self._controller.get_all_spectra():
            metadata = self._controller.get_metadata(spectrum_id)

            file_attr = getattr(metadata, "file", None)
            group_attr = getattr(metadata, "group", None)
            name_attr = getattr(metadata, "name", None)

            grouped[file_attr][group_attr].append((name_attr, spectrum_id))

        for file, groups in sorted(grouped.items()):
            file_item = SpectrumTreeItem(
                _label=file,
                label=file.split("/")[-1] or "No file",
                kind="file",
                parent=self._root_item,
            )
            self._root_item.append_child(file_item)

            for group, spectra in sorted(groups.items()):
                group_item = SpectrumTreeItem(
                    kind="group",
                    parent=file_item,
                    _label=group,
                    label=group or "No group",
                )
                file_item.append_child(group_item)

                for name, spectrum_id in sorted(spectra):
                    label_name = name or "No name"
                    label_name = (
                        f"{label_name} {spectrum_id[:5]}" if params.show_spectrum_id_in_tree else label_name
                    )
                    spectrum_item = SpectrumTreeItem(
                        _label=name,
                        label=label_name,
                        kind="spectrum",
                        parent=group_item,
                        spectrum_id=spectrum_id,
                    )

                    group_item.append_child(spectrum_item)

        self.endResetModel()

    def item_from_index(self, index: QModelIndex) -> Optional[SpectrumTreeItem]:
        """
        Return the :class:`SpectrumTreeItem` associated with an index.

        Parameters
        ----------
        index : QModelIndex
            Model index obtained from a connected view.

        Returns
        -------
        SpectrumTreeItem or None
            Backing item instance, if any.
        """
        if not index.isValid():
            return None
        item = index.internalPointer()
        if isinstance(item, SpectrumTreeItem):
            return item
        return None


class SpectrumTreeWidget(QTreeView):
    """
    View for the spectrum tree hierarchy.

    This widget owns a :class:`SpectrumTreeModel` instance and forwards
    selection changes to the provided :class:`ControllerWrapper` so that the
    rest of the UI can react via Qt signals.
    """

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._model = SpectrumTreeModel(controller, self)
        self.setModel(self._model)
        self.setHeaderHidden(True)
        self.setSelectionMode(QTreeView.ExtendedSelection)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu_requested)

        selection_model = self.selectionModel()
        selection_model.selectionChanged.connect(self._on_selection_changed)

    @property
    def model(self) -> SpectrumTreeModel:
        """
        Return the underlying spectrum tree model.
        """
        return self._model

    def refresh(self) -> None:
        """
        Refresh tree contents from the controller and expand top-level items.
        """
        self._model.refresh()
        self.expandAll()

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def get_selected_items(self) -> list[SpectrumTreeItem]:
        """
        Return unique tree items corresponding to the current selection.

        Returns
        -------
        list[SpectrumTreeItem]
            Unique items backing the currently selected indexes.
        """
        items: list[SpectrumTreeItem] = []
        for index in self.selectionModel().selectedIndexes():
            item = self._model.item_from_index(index)
            if item is not None and item not in items:
                items.append(item)
        return items

    def get_selected_spectrum_ids(self) -> list[str]:
        """
        Return spectrum identifiers represented by the current selection.

        File and group items are expanded to all spectra contained within
        their subtree. Spectrum items contribute their own ``spectrum_id``.

        Returns
        -------
        list[str]
            Unique spectrum identifiers covered by the selection.
        """

        def collect_spectra(item: SpectrumTreeItem, acc: set[str]) -> None:
            if item.kind == "spectrum" and item.spectrum_id is not None:
                acc.add(item.spectrum_id)
            for child in item.children:
                collect_spectra(child, acc)

        selected_items = self.get_selected_items()
        spectrum_ids: set[str] = set()
        for item in selected_items:
            collect_spectra(item, spectrum_ids)
        return sorted(spectrum_ids)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selection_changed(self, selected: Any, _deselected: Any) -> None:
        """
        Forward spectrum selection to the controller wrapper.

        Parameters
        ----------
        selected : QItemSelection
            Newly selected indexes.
        _deselected : QItemSelection
            No longer selected indexes (unused).
        """
        del _deselected

        indexes = selected.indexes()
        if not indexes:
            self._controller.set_selection(None, None)
            return

        # Preserve single-selection semantics for the rest of the UI by
        # forwarding only the first spectrum in the current selection.
        index = indexes[0]
        item = self._model.item_from_index(index)
        if item is None or item.kind != "spectrum" or item.spectrum_id is None:
            self._controller.set_selection(None, None)
            return

        self._controller.set_selection(item.spectrum_id, None)

    def _on_context_menu_requested(self, pos: QPoint) -> None:
        """
        Show context menu for rename/delete/copy operations.

        Parameters
        ----------
        pos : QPoint
            Position in viewport coordinates where the menu was requested.
        """
        index = self.indexAt(pos)
        if not index.isValid():
            return

        item = self._model.item_from_index(index)
        if item is None:
            return

        menu = QMenu(self)

        rename_action = None
        delete_action = None
        copy_id_action = None

        if item.kind == "spectrum":
            rename_action = menu.addAction("Rename spectrum")
            delete_action = menu.addAction("Delete spectrum")
            if item.spectrum_id is not None:
                copy_id_action = menu.addAction("Copy spectrum ID")
        elif item.kind == "group":
            rename_action = menu.addAction("Rename group")
            delete_action = menu.addAction("Delete group")
        elif item.kind == "file":
            rename_action = menu.addAction("Rename file")
            delete_action = menu.addAction("Delete file")

        if menu.isEmpty():
            return

        chosen = menu.exec(self.viewport().mapToGlobal(pos))
        if chosen is None:
            return

        if chosen is rename_action:
            self._handle_rename(item)
        elif chosen is delete_action:
            self._handle_delete(item)
        elif chosen is copy_id_action:
            self._handle_copy_id(item)

    def _handle_rename(self, item: SpectrumTreeItem) -> None:
        """
        Handle rename action for spectrum, group, or file items.

        Parameters
        ----------
        item : SpectrumTreeItem
            Tree item to rename.
        """

        new_label, ok = QInputDialog.getText(self, "Rename", "New name:", text=item._label)
        if not ok:
            return
        new_label = new_label.strip()
        if not new_label or new_label == item._label:
            return

        if item.kind == "spectrum" and item.spectrum_id is not None:
            self._controller.rename_spectrum(item.spectrum_id, new_label)
        elif item.kind == "group":
            parent = item.parent
            if parent is None or parent.kind != "file":
                return
            file_label = parent._label
            self._controller.rename_group(file_label, item._label, new_label)
        elif item.kind == "file":
            self._controller.rename_file(item._label, new_label)

        self.refresh()

    def _handle_delete(self, item: SpectrumTreeItem) -> None:
        """
        Handle delete action for spectrum, group, or file items.

        Parameters
        ----------
        item : SpectrumTreeItem
            Tree item to delete.
        """
        if item.kind == "spectrum" and item.spectrum_id is not None:
            self._controller.full_remove_object(item.spectrum_id)
        elif item.kind == "group":
            parent = item.parent
            if parent is None or parent.kind != "file":
                return
            file_label = parent._label
            self._controller.remove_group(file_label, item._label)
        elif item.kind == "file":
            self._controller.remove_file(item._label)

        self.refresh()

    def _handle_copy_id(self, item: SpectrumTreeItem) -> None:
        """
        Copy the spectrum identifier of a spectrum item to the clipboard.

        Parameters
        ----------
        item : SpectrumTreeItem
            Spectrum item whose ID should be copied.
        """
        if item.spectrum_id is None:
            return
        QApplication.clipboard().setText(item.spectrum_id)
