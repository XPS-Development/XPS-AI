from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from PySide6.QtCore import QAbstractItemModel, QModelIndex, QPoint, Qt
from PySide6.QtWidgets import QApplication, QInputDialog, QMenu, QTreeView, QWidget

from .controller import ControllerWrapper
from .export_options_dialog import export_peaks, export_spectra


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
        Refresh tree contents from the controller while preserving expand/collapse state.
        """
        expanded = self._collect_expanded_stable_keys()
        self._model.refresh()
        self._restore_expanded_stable_keys(expanded)

    @staticmethod
    def _stable_key_for_item(item: SpectrumTreeItem) -> tuple[Any, ...]:
        """Return a key stable across model rebuilds for the given tree item."""
        if item.kind == "file":
            return ("file", item._label)
        if item.kind == "group":
            parent = item.parent
            file_l = parent._label if parent is not None and parent.kind == "file" else None
            return ("group", file_l, item._label)
        if item.kind == "spectrum":
            g_parent = item.parent
            file_l = None
            group_l = None
            if g_parent is not None and g_parent.kind == "group":
                group_l = g_parent._label
                f_parent = g_parent.parent
                if f_parent is not None and f_parent.kind == "file":
                    file_l = f_parent._label
            return ("spectrum", file_l, group_l, item.spectrum_id)
        return ("other", item.kind, item._label)

    def _collect_expanded_stable_keys(self) -> set[tuple[Any, ...]]:
        """Return stable keys for all expanded nodes in the tree."""
        keys: set[tuple[Any, ...]] = set()

        def walk(parent: QModelIndex) -> None:
            for row in range(self._model.rowCount(parent)):
                idx = self._model.index(row, 0, parent)
                if self.isExpanded(idx):
                    item = self._model.item_from_index(idx)
                    if item is not None:
                        keys.add(self._stable_key_for_item(item))
                walk(idx)

        walk(QModelIndex())
        return keys

    def _restore_expanded_stable_keys(self, keys: set[tuple[Any, ...]]) -> None:
        """Expand nodes whose stable keys match a previously expanded set."""

        def walk(parent: QModelIndex) -> None:
            for row in range(self._model.rowCount(parent)):
                idx = self._model.index(row, 0, parent)
                item = self._model.item_from_index(idx)
                if item is not None and self._stable_key_for_item(item) in keys:
                    self.setExpanded(idx, True)
                walk(idx)

        walk(QModelIndex())

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _collect_spectra_in_subtree(self, item: SpectrumTreeItem, acc: list[str], seen: set[str]) -> None:
        """
        Collect spectrum identifiers from the subtree rooted at ``item``.

        The traversal follows the tree order so that the first collected
        spectrum can be used as a stable primary selection.

        Parameters
        ----------
        item : SpectrumTreeItem
            Tree item whose subtree should be traversed.
        acc : list[str]
            List used to accumulate spectrum identifiers in order.
        seen : set[str]
            Set of already collected identifiers used to preserve uniqueness.
        """
        if item.kind == "spectrum" and item.spectrum_id is not None and item.spectrum_id not in seen:
            seen.add(item.spectrum_id)
            acc.append(item.spectrum_id)
        for child in item.children:
            self._collect_spectra_in_subtree(child, acc, seen)

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
            Unique spectrum identifiers covered by the selection, in a stable
            tree order suitable for determining a primary spectrum.
        """
        selected_items = self.get_selected_items()
        spectrum_ids: list[str] = []
        seen: set[str] = set()
        for item in selected_items:
            self._collect_spectra_in_subtree(item, spectrum_ids, seen)
        return spectrum_ids

    def get_primary_spectrum_id(self) -> str | None:
        """
        Return the primary spectrum identifier for the current selection.

        The primary spectrum is defined as the first spectrum identifier in the
        ordered list returned by :meth:`get_selected_spectrum_ids`.

        Returns
        -------
        str or None
            Primary spectrum identifier or None if no spectra are selected.
        """
        spectrum_ids = self.get_selected_spectrum_ids()
        if not spectrum_ids:
            return None
        return spectrum_ids[0]

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selection_changed(self, _selected: Any, _deselected: Any) -> None:
        """
        Forward spectrum selection to the controller wrapper.

        Parameters
        ----------
        selected : QItemSelection
            Newly selected indexes.
        _deselected : QItemSelection
            No longer selected indexes (unused).
        """
        del _selected, _deselected

        # Preserve single-spectrum semantics for the rest of the UI by
        # forwarding only the primary spectrum from the current (possibly
        # multi-)selection.
        primary_spectrum_id = self.get_primary_spectrum_id()
        self._controller.set_selection(primary_spectrum_id, None)

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

        if item.kind == "spectrum":
            rename_action = menu.addAction("Rename spectrum")
            rename_action.triggered.connect(lambda: self._handle_rename(item))
            delete_action = menu.addAction("Delete spectrum")
            delete_action.triggered.connect(lambda: self._handle_delete(item))
            if item.spectrum_id is not None:
                copy_id_action = menu.addAction("Copy spectrum ID")
                copy_id_action.triggered.connect(lambda: self._handle_copy_id(item))
                export_menu = menu.addMenu("Export...")
                export_spectrum_action = export_menu.addAction("Export spectrum")
                export_spectrum_action.triggered.connect(lambda: self._handle_export_spectrum(item))
                export_peaks_action = export_menu.addAction("Export peaks")
                export_peaks_action.triggered.connect(lambda: self._handle_export_peaks(item))
                export_all_spectra_action = export_menu.addAction("Export all selected spectra")
                export_all_spectra_action.triggered.connect(self._handle_export_all_selected_spectra)
                export_all_peaks_action = export_menu.addAction("Export peaks from all selected spectra")
                export_all_peaks_action.triggered.connect(self._handle_export_peaks_all_selected_spectra)
        elif item.kind == "group":
            rename_action = menu.addAction("Rename group")
            rename_action.triggered.connect(lambda: self._handle_rename(item))
            delete_action = menu.addAction("Delete group")
            delete_action.triggered.connect(lambda: self._handle_delete(item))
            export_menu = menu.addMenu("Export...")
            export_all_spectra_action = export_menu.addAction("Export all selected spectra")
            export_all_spectra_action.triggered.connect(self._handle_export_all_selected_spectra)
            export_all_peaks_action = export_menu.addAction("Export peaks from all selected spectra")
            export_all_peaks_action.triggered.connect(self._handle_export_peaks_all_selected_spectra)
        elif item.kind == "file":
            rename_action = menu.addAction("Rename file")
            rename_action.triggered.connect(lambda: self._handle_rename(item))
            delete_action = menu.addAction("Delete file")
            delete_action.triggered.connect(lambda: self._handle_delete(item))
            export_menu = menu.addMenu("Export...")
            export_all_spectra_action = export_menu.addAction("Export all selected spectra")
            export_all_spectra_action.triggered.connect(self._handle_export_all_selected_spectra)
            export_all_peaks_action = export_menu.addAction("Export peaks from all selected spectra")
            export_all_peaks_action.triggered.connect(self._handle_export_peaks_all_selected_spectra)
        if menu.isEmpty():
            return

        menu.exec(self.viewport().mapToGlobal(pos))

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

    def _handle_export_spectrum(self, item: SpectrumTreeItem) -> None:
        """
        Export a spectrum item into a CSV-like file.

        Parameters
        ----------
        item : SpectrumTreeItem
            Spectrum item to export.
        """
        if item.spectrum_id is None:
            return
        export_spectra(self._controller, [item.spectrum_id], parent=self)

    def _handle_export_peaks(self, item: SpectrumTreeItem) -> None:
        """
        Export peak parameters of spectrum item.
        """
        if item.spectrum_id is None:
            return
        export_peaks(self._controller, [item.spectrum_id], parent=self)

    def _handle_export_all_selected_spectra(self) -> None:
        spectrum_ids = self.get_selected_spectrum_ids()
        if not spectrum_ids:
            return
        export_spectra(self._controller, spectrum_ids, parent=self)

    def _handle_export_peaks_all_selected_spectra(self) -> None:
        spectrum_ids = self.get_selected_spectrum_ids()
        if not spectrum_ids:
            return
        export_peaks(self._controller, spectrum_ids, parent=self)
