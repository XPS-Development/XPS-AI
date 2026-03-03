from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt
from PySide6.QtWidgets import QTreeView, QWidget

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

    label: str
    kind: str
    parent: Optional["SpectrumTreeItem"] = None
    spectrum_id: Optional[str] = None
    children: list["SpectrumTreeItem"] = field(default_factory=list)

    def child(self, row: int) -> Optional["SpectrumTreeItem"]:
        """Return child at the given row index."""
        if 0 <= row < len(self.children):
            return self.children[row]
        return None

    def row(self) -> int:
        """Return the index of this item within its parent."""
        if self.parent is None:
            return 0
        return self.parent.children.index(self)

    def append_child(self, item: "SpectrumTreeItem") -> None:
        """Append a child item to this node."""
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
        self._root_item = SpectrumTreeItem(label="Root", kind="root")
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
        """
        Rebuild the internal tree structure from controller data.

        This method queries all spectra and their metadata from the wrapped
        :class:`ControllerWrapper` and emits a full model reset so that any
        attached views update their contents.
        """
        self.beginResetModel()
        self._root_item.children.clear()

        spectra: Iterable[str] = self._controller.get_all_spectra()

        # Build nested mapping: file -> group -> list[spectrum_id]
        grouped: dict[str, dict[str, list[str]]] = {}
        for spectrum_id in spectra:
            metadata = self._controller.get_metadata(spectrum_id)
            if metadata is not None:
                file_label = str(getattr(metadata, "file", "No file"))
                if not file_label:
                    file_label = "No file"
                group_label = str(getattr(metadata, "group", "No group"))
                if not group_label:
                    group_label = "No group"
                name_label = str(getattr(metadata, "name", "No name"))
                if not name_label:
                    name_label = "No name"

            file_groups = grouped.setdefault(file_label, {})
            spectra_list = file_groups.setdefault(group_label, [])
            spectra_list.append((spectrum_id, name_label))

        # Create items
        for file_label, groups in sorted(grouped.items(), key=lambda kv: kv[0]):
            file_item = SpectrumTreeItem(label=file_label, kind="file", parent=self._root_item)
            self._root_item.append_child(file_item)
            for group_label, spectrum_ids in sorted(groups.items(), key=lambda kv: kv[0]):
                group_item = SpectrumTreeItem(label=group_label, kind="group", parent=file_item)
                file_item.append_child(group_item)
                for spectrum_id, name_label in sorted(spectrum_ids):
                    spectrum_label = f"{name_label} {spectrum_id[:5]}"
                    spectrum_item = SpectrumTreeItem(
                        label=spectrum_label,
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

        selection_model = self.selectionModel()
        selection_model.selectionChanged.connect(self._on_selection_changed)

    def model(self) -> SpectrumTreeModel:  # type: ignore[override]
        """
        Return the underlying :class:`SpectrumTreeModel` instance.

        Returns
        -------
        SpectrumTreeModel
            Backing model used by this view.
        """
        return self._model

    def refresh(self) -> None:
        """
        Refresh tree contents from the controller and expand top-level items.
        """
        self._model.refresh()
        self.expandAll()

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

        index = indexes[0]
        item = self._model.item_from_index(index)
        if item is None or item.kind != "spectrum" or item.spectrum_id is None:
            self._controller.set_selection(None, None)
            return

        self._controller.set_selection(item.spectrum_id, None)
