"""Dialog to copy a spectrum decomposition onto other spectra."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from PySide6.QtCore import (
    QAbstractItemModel,
    QEvent,
    QModelIndex,
    QPersistentModelIndex,
    QRect,
    QSize,
    Qt,
)
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .component_colors import (
    STATUS_COLOR_EMPTY,
    STATUS_COLOR_PEAKS,
    STATUS_COLOR_REGIONS,
    color_for_component,
)
from .name_id_delegate import (
    ComponentColorRole,
    NameWithIdDelegate,
    ObjectIdPrefixRole,
    ObjectIdRole,
)
from .optimize_confirm import confirm_and_optimize
from .tree_style import EditorTreeView, apply_editor_tree_style

if TYPE_CHECKING:
    from .controller import ControllerWrapper

_DEFAULT_INDEX = QModelIndex()
_ID_DISPLAY_CHARS = 5

_STATUS_COLORS = {
    "empty": STATUS_COLOR_EMPTY,
    "regions": STATUS_COLOR_REGIONS,
    "peaks": STATUS_COLOR_PEAKS,
}
_STATUS_RANK = {"empty": 0, "regions": 1, "peaks": 2}


@dataclass
class _TreeItem:
    """Generic tree node for the copy dialog models."""

    label: str
    kind: str
    parent: _TreeItem | None = None
    object_id: str | None = None
    param_name: str | None = None
    component_id: str | None = None
    component_kind: str | None = None
    search_text: str = ""
    linked: bool = False
    _row: int = 0
    children: list[_TreeItem] = field(default_factory=list)

    def child(self, row: int) -> _TreeItem | None:
        if 0 <= row < len(self.children):
            return self.children[row]
        return None

    def append_child(self, item: _TreeItem) -> None:
        item.parent = self
        item._row = len(self.children)
        self.children.append(item)


class _BaseTreeModel(QAbstractItemModel):
    """Shared QAbstractItemModel helpers for the copy dialog trees."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._root = _TreeItem(label="Root", kind="root")

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX) -> int:
        item = self._root if not parent.isValid() else parent.internalPointer()
        if not isinstance(item, _TreeItem):
            return 0
        return len(item.children)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX) -> int:
        del parent
        return 1

    def index(
        self,
        row: int,
        column: int,
        parent: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX,
    ) -> QModelIndex:
        if column < 0 or row < 0 or column >= self.columnCount(parent):
            return QModelIndex()
        parent_item = self._root if not parent.isValid() else parent.internalPointer()
        if not isinstance(parent_item, _TreeItem):
            return QModelIndex()
        child = parent_item.child(row)
        if child is None:
            return QModelIndex()
        return self.createIndex(row, column, child)

    def parent(  # ty: ignore[invalid-method-override]
        self, child: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX
    ) -> QModelIndex:
        """Return the parent model index of ``child``."""
        if not child.isValid():
            return QModelIndex()
        item = child.internalPointer()
        if not isinstance(item, _TreeItem) or item.parent is None or item.parent is self._root:
            return QModelIndex()
        parent_item = item.parent
        return self.createIndex(parent_item._row, 0, parent_item)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """Return display or object-id data for ``index``."""
        if not index.isValid() or index.column() != 0:
            return None
        item = index.internalPointer()
        if not isinstance(item, _TreeItem):
            return None
        if role in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
            return item.label
        if role == ObjectIdRole and item.object_id is not None:
            return item.object_id
        if role == ObjectIdPrefixRole and item.object_id is not None:
            return item.object_id[:_ID_DISPLAY_CHARS]
        return None

    def flags(self, index: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX) -> Qt.ItemFlag:
        """Return selectable/enabled flags for ``index``."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


class TargetSpectrumModel(_BaseTreeModel):
    """File → group → spectrum tree for multi-select copy targets."""

    def __init__(
        self,
        controller: ControllerWrapper,
        *,
        source_spectrum_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._source_spectrum_id = source_spectrum_id
        self.refresh()

    def refresh(self) -> None:
        """Rebuild spectra hierarchy, excluding the source spectrum."""
        self.beginResetModel()
        self._root.children.clear()
        query = self._controller.query
        grouped: dict[Any, dict[Any, list[tuple[Any, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for spectrum_id in query.get_all_spectra_ids():
            if spectrum_id == self._source_spectrum_id:
                continue
            metadata = query.get_metadata(spectrum_id)
            file_attr = getattr(metadata, "file", None)
            group_attr = getattr(metadata, "group", None)
            name_attr = getattr(metadata, "name", None)
            grouped[file_attr][group_attr].append((name_attr, spectrum_id))

        for file_key, groups in sorted(grouped.items(), key=lambda kv: str(kv[0] or "")):
            file_label = (str(file_key).split("/")[-1] if file_key else "No file") or "No file"
            file_item = _TreeItem(
                label=file_label,
                kind="file",
                search_text=file_label.lower(),
            )
            self._root.append_child(file_item)
            for group_key, spectra in sorted(groups.items(), key=lambda kv: str(kv[0] or "")):
                group_label = str(group_key) if group_key else "No group"
                group_item = _TreeItem(
                    label=group_label,
                    kind="group",
                    search_text=group_label.lower(),
                )
                file_item.append_child(group_item)
                for name, spectrum_id in sorted(spectra, key=lambda t: str(t[0] or "")):
                    spectrum_label = str(name) if name else "No name"
                    group_item.append_child(
                        _TreeItem(
                            label=spectrum_label,
                            kind="spectrum",
                            object_id=spectrum_id,
                            search_text=f"{spectrum_label} {spectrum_id}".lower(),
                        )
                    )
        self.endResetModel()

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """Return labels, short ids, and structure-status colors."""
        if role == ComponentColorRole and index.isValid():
            item = index.internalPointer()
            if isinstance(item, _TreeItem) and item.kind in {"file", "group", "spectrum"}:
                return _STATUS_COLORS.get(self._structure_status(item))
            return None
        if role == ObjectIdPrefixRole and index.isValid():
            item = index.internalPointer()
            if isinstance(item, _TreeItem) and item.kind == "spectrum" and item.object_id:
                return item.object_id[:_ID_DISPLAY_CHARS]
            return None
        return super().data(index, role)

    def flags(self, index: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX) -> Qt.ItemFlag:
        """Allow selecting spectrum leaves only."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        item = index.internalPointer()
        base = Qt.ItemFlag.ItemIsEnabled
        if isinstance(item, _TreeItem) and item.kind == "spectrum":
            return base | Qt.ItemFlag.ItemIsSelectable
        return base

    def spectrum_ids_under(self, index: QModelIndex) -> list[str]:
        """Return spectrum object ids in the subtree of ``index``."""
        if not index.isValid():
            return []
        item = index.internalPointer()
        if not isinstance(item, _TreeItem):
            return []
        return self._spectrum_ids_in_subtree(item)

    def _structure_status(self, item: _TreeItem) -> str:
        query = self._controller.query
        if item.kind == "spectrum" and item.object_id is not None:
            return query.get_spectrum_structure_status(item.object_id)
        if item.kind in {"file", "group"}:
            best = "empty"
            for sid in self._spectrum_ids_in_subtree(item):
                status = query.get_spectrum_structure_status(sid)
                if _STATUS_RANK[status] > _STATUS_RANK[best]:
                    best = status
            return best
        return "empty"

    def _spectrum_ids_in_subtree(self, item: _TreeItem) -> list[str]:
        found: list[str] = []

        def walk(node: _TreeItem) -> None:
            if node.kind == "spectrum" and node.object_id is not None:
                found.append(node.object_id)
            for child in node.children:
                walk(child)

        walk(item)
        return found


class LinkTreeModel(_BaseTreeModel):
    """
    Source spectrum Region → component → parameter tree with a Link column.

    Column 0 shows name / color / short id. Column 1 holds link checkboxes.
    Region and component checkboxes are tri-state; parameters are binary.
    """

    def __init__(
        self,
        controller: ControllerWrapper,
        *,
        source_spectrum_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._source_spectrum_id = source_spectrum_id
        self.refresh()

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX) -> int:
        """Return two columns: name and link checkbox."""
        del parent
        return 2

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """Return ``Name`` / ``Link`` headers."""
        if orientation != Qt.Orientation.Horizontal or role != Qt.ItemDataRole.DisplayRole:
            return None
        if section == 0:
            return "Name"
        if section == 1:
            return "Link"
        return None

    def refresh(self) -> None:
        """Rebuild the link tree from the source spectrum."""
        self.beginResetModel()
        self._root.children.clear()
        query = self._controller.query
        for region_index, region_id in enumerate(
            query.get_regions_ids(self._source_spectrum_id), start=1
        ):
            region_item = _TreeItem(
                label=f"Region {region_index}",
                kind="region",
                object_id=region_id,
                search_text=f"region {region_index} {region_id}".lower(),
            )
            self._root.append_child(region_item)

            bg_id = query.get_background_id(region_id)
            if bg_id is not None:
                bg_dto = query.get_component_dto(bg_id, normalized=False)
                self._append_component(
                    region_item,
                    bg_dto.id_,
                    bg_dto.name or "Background",
                    bg_dto,
                    component_kind="background",
                )

            for peak_index, peak_id in enumerate(query.get_peaks_ids(region_id), start=1):
                peak_dto = query.get_component_dto(peak_id, normalized=False)
                label = peak_dto.name or f"Peak {peak_index}"
                self._append_component(
                    region_item,
                    peak_id,
                    label,
                    peak_dto,
                    component_kind="peak",
                )
        self.endResetModel()

    def _append_component(
        self,
        region_item: _TreeItem,
        component_id: str,
        label: str,
        dto: Any,
        *,
        component_kind: str,
    ) -> None:
        component_item = _TreeItem(
            label=label,
            kind="component",
            object_id=component_id,
            component_id=component_id,
            component_kind=component_kind,
            search_text=f"{label} {component_id}".lower(),
        )
        region_item.append_child(component_item)
        for name in dto.parameters:
            component_item.append_child(
                _TreeItem(
                    label=name,
                    kind="parameter",
                    object_id=component_id,
                    component_id=component_id,
                    param_name=name,
                    search_text=name.lower(),
                    linked=False,
                )
            )

    def flags(self, index: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX) -> Qt.ItemFlag:
        """Name column selectable; Link column checkable."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        if index.column() == 1:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """Return name/color/id for column 0 and check-state for column 1."""
        if not index.isValid():
            return None
        item = index.internalPointer()
        if not isinstance(item, _TreeItem):
            return None

        if index.column() == 1:
            if role == Qt.ItemDataRole.CheckStateRole:
                return self._check_state(item)
            if role == Qt.ItemDataRole.DisplayRole:
                return ""
            return None

        if role == ComponentColorRole and item.kind == "component":
            return color_for_component(
                kind=item.component_kind or "peak",
                component_id=item.component_id,
            )
        if role == ObjectIdPrefixRole and item.object_id is not None:
            if item.kind in {"region", "component"}:
                return item.object_id[:_ID_DISPLAY_CHARS]
            return None
        if role == ObjectIdRole and item.object_id is not None and item.kind != "parameter":
            return item.object_id
        if role in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
            return item.label
        return None

    def setData(
        self,
        index: QModelIndex | QPersistentModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        """Cascade link checkboxes from region/component/parameter nodes."""
        if role != Qt.ItemDataRole.CheckStateRole or not index.isValid() or index.column() != 1:
            return False
        item = index.internalPointer()
        if not isinstance(item, _TreeItem):
            return False
        linked = value == Qt.CheckState.Checked or value == Qt.CheckState.Checked.value
        self._set_linked_cascade(item, linked)
        self._emit_check_branch(index)
        return True

    def link_flags(self) -> dict[tuple[str, str], bool]:
        """Return ``(component_id, param_name) -> linked`` for all parameters."""
        flags: dict[tuple[str, str], bool] = {}

        def walk(node: _TreeItem) -> None:
            if (
                node.kind == "parameter"
                and node.component_id is not None
                and node.param_name is not None
            ):
                flags[(node.component_id, node.param_name)] = node.linked
            for child in node.children:
                walk(child)

        walk(self._root)
        return flags

    def clear_links(self) -> None:
        """Uncheck every link checkbox."""
        self.beginResetModel()

        def walk(node: _TreeItem) -> None:
            node.linked = False
            for child in node.children:
                walk(child)

        walk(self._root)
        self.endResetModel()

    def _check_state(self, item: _TreeItem) -> Qt.CheckState:
        if item.kind == "parameter":
            return Qt.CheckState.Checked if item.linked else Qt.CheckState.Unchecked
        if not item.children:
            return Qt.CheckState.Unchecked
        states = [self._check_state(child) for child in item.children]
        if all(s == Qt.CheckState.Checked for s in states):
            return Qt.CheckState.Checked
        if all(s == Qt.CheckState.Unchecked for s in states):
            return Qt.CheckState.Unchecked
        return Qt.CheckState.PartiallyChecked

    def _set_linked_cascade(self, item: _TreeItem, linked: bool) -> None:
        item.linked = linked
        for child in item.children:
            self._set_linked_cascade(child, linked)

    def _emit_check_branch(self, index: QModelIndex | QPersistentModelIndex) -> None:
        link_index = index.sibling(index.row(), 1)
        self.dataChanged.emit(link_index, link_index, [Qt.ItemDataRole.CheckStateRole])
        parent = index.parent()
        while parent.isValid():
            parent_link = parent.sibling(parent.row(), 1)
            self.dataChanged.emit(parent_link, parent_link, [Qt.ItemDataRole.CheckStateRole])
            parent = parent.parent()
        self._emit_descendants(index.sibling(index.row(), 0))

    def _emit_descendants(self, index: QModelIndex | QPersistentModelIndex) -> None:
        rows = self.rowCount(index)
        for row in range(rows):
            child = self.index(row, 1, index)
            self.dataChanged.emit(child, child, [Qt.ItemDataRole.CheckStateRole])
            self._emit_descendants(self.index(row, 0, index))


class LinkCheckboxDelegate(QStyledItemDelegate):
    """
    Center a themed checkbox in the Link column.

    Uses :func:`theme.paint_rounded_checkbox` via the application style so
    indicators match ``QCheckBox`` and other item-view checks.
    """

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Paint row chrome, then a centered themed check indicator."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        opt.features &= ~QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
        widget = opt.widget
        style = widget.style() if widget is not None else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)

        state = index.data(Qt.ItemDataRole.CheckStateRole)
        check = QStyleOptionButton()
        check.rect = self._indicator_rect(option.rect)
        check.state = QStyle.StateFlag.State_Enabled
        if option.state & QStyle.StateFlag.State_MouseOver:
            check.state |= QStyle.StateFlag.State_MouseOver
        if state in {Qt.CheckState.Checked, Qt.CheckState.Checked.value}:
            check.state |= QStyle.StateFlag.State_On
        elif state in {Qt.CheckState.PartiallyChecked, Qt.CheckState.PartiallyChecked.value}:
            check.state |= QStyle.StateFlag.State_NoChange
        else:
            check.state |= QStyle.StateFlag.State_Off
        if style is not None:
            style.drawPrimitive(
                QStyle.PrimitiveElement.PE_IndicatorItemViewItemCheck,
                check,
                painter,
                widget,
            )

    def sizeHint(
        self,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> QSize:
        """Ensure the Link column is wide enough for the checkbox."""
        del index
        return QSize(48, max(option.rect.height(), theme.CHECKBOX_SIZE + 8))

    def editorEvent(
        self,
        event: QEvent,
        model: QAbstractItemModel,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> bool:
        """Toggle link state on left-click or Space in the Link cell."""
        if not index.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            return False
        toggle = False
        if event.type() in {
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.MouseButtonDblClick,
        } and isinstance(event, QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton and option.rect.contains(
                event.position().toPoint()
            ):
                toggle = True
        elif event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if event.key() in {Qt.Key.Key_Space, Qt.Key.Key_Select}:
                toggle = True
        if not toggle:
            return False
        current = index.data(Qt.ItemDataRole.CheckStateRole)
        checked = current in {Qt.CheckState.Checked, Qt.CheckState.Checked.value}
        new_state = Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked
        return model.setData(index, new_state, Qt.ItemDataRole.CheckStateRole)

    def _indicator_rect(self, cell: QRect) -> QRect:
        box = theme.CHECKBOX_SIZE
        return QRect(
            cell.x() + max(0, (cell.width() - box) // 2),
            cell.y() + max(0, (cell.height() - box) // 2),
            box,
            box,
        )


def _apply_filter(view: EditorTreeView, model: _BaseTreeModel, text: str) -> None:
    """Hide rows that do not match ``text`` (substring on ``search_text``)."""
    needle = text.strip().lower()

    def match(item: _TreeItem) -> bool:
        if not needle:
            return True
        if needle in item.search_text or needle in item.label.lower():
            return True
        return any(match(child) for child in item.children)

    def apply(parent: QModelIndex, item: _TreeItem) -> bool:
        visible = match(item) if parent.isValid() else True
        any_visible = False
        for row, child in enumerate(item.children):
            child_index = model.index(row, 0, parent)
            child_visible = apply(child_index, child)
            view.setRowHidden(row, parent, not child_visible)
            any_visible = any_visible or child_visible
        if parent.isValid():
            return visible or any_visible
        return True

    apply(QModelIndex(), model._root)
    if needle:
        view.expandAll()


class CopyDecompositionDialog(QDialog):
    """
    Copy the selected spectrum's regions/peaks onto other spectra.

    Non-modal: the main window stays usable. Left: multi-select target spectra
    with status dots. Right: link checkboxes (dedicated column) with component
    colors. Options rescale intensities and optimize after copy.
    """

    def __init__(
        self,
        controller: ControllerWrapper,
        source_spectrum_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Copy decomposition")
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(860, 520)
        self._controller = controller
        self._source_spectrum_id = source_spectrum_id

        self._target_model = TargetSpectrumModel(
            controller, source_spectrum_id=source_spectrum_id, parent=self
        )
        self._link_model = LinkTreeModel(
            controller, source_spectrum_id=source_spectrum_id, parent=self
        )

        self._target_search = QLineEdit(self)
        self._target_search.setPlaceholderText("Search spectra…")
        self._target_view = EditorTreeView(self)
        apply_editor_tree_style(self._target_view)
        self._target_view.setModel(self._target_model)
        self._target_view.setHeaderHidden(True)
        self._target_view.setSelectionMode(EditorTreeView.SelectionMode.ExtendedSelection)
        self._target_view.setItemDelegate(NameWithIdDelegate(self._target_view))
        self._target_view.expandAll()

        self._link_search = QLineEdit(self)
        self._link_search.setPlaceholderText("Search parameters…")
        self._link_view = EditorTreeView(self)
        apply_editor_tree_style(self._link_view)
        self._link_view.setModel(self._link_model)
        self._link_view.setHeaderHidden(False)
        self._link_view.setItemDelegateForColumn(0, NameWithIdDelegate(self._link_view))
        self._link_view.setItemDelegateForColumn(1, LinkCheckboxDelegate(self._link_view))
        header = self._link_view.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 48)
        self._link_view.expandAll()

        self._rescale_cb = QCheckBox("Rescale intensities", self)
        self._rescale_cb.setChecked(True)
        self._optimize_cb = QCheckBox("Optimize after copy", self)
        self._optimize_cb.setChecked(True)

        left = QVBoxLayout()
        left.addWidget(QLabel("Copy to spectra", self))
        left.addWidget(self._target_search)
        left.addWidget(self._target_view)
        left_w = QWidget(self)
        left_w.setLayout(left)

        right = QVBoxLayout()
        right.addWidget(QLabel("Link parameters to source", self))
        right.addWidget(self._link_search)
        right.addWidget(self._link_view)
        right_w = QWidget(self)
        right_w.setLayout(right)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        options = QHBoxLayout()
        options.addWidget(self._rescale_cb)
        options.addWidget(self._optimize_cb)
        options.addStretch(1)

        buttons = QDialogButtonBox(self)
        self._close_btn = buttons.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        self._reset_btn = buttons.addButton(
            "Reset selection", QDialogButtonBox.ButtonRole.ActionRole
        )
        self._copy_btn = buttons.addButton("Copy", QDialogButtonBox.ButtonRole.AcceptRole)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        layout.addLayout(options)
        layout.addWidget(buttons)

        self._close_btn.clicked.connect(self.close)
        self._reset_btn.clicked.connect(self._on_reset)
        self._copy_btn.clicked.connect(self._on_copy)
        self._target_search.textChanged.connect(self._on_target_filter)
        self._link_search.textChanged.connect(self._on_link_filter)

    def _on_target_filter(self, text: str) -> None:
        _apply_filter(self._target_view, self._target_model, text)

    def _on_link_filter(self, text: str) -> None:
        _apply_filter(self._link_view, self._link_model, text)

    def _on_reset(self) -> None:
        self._target_view.clearSelection()
        self._link_model.clear_links()
        self._link_view.expandAll()
        self._rescale_cb.setChecked(True)
        self._optimize_cb.setChecked(True)

    def _selected_target_ids(self) -> list[str]:
        ids: list[str] = []
        for index in self._target_view.selectionModel().selectedIndexes():
            ids.extend(self._target_model.spectrum_ids_under(index))
        seen: set[str] = set()
        ordered: list[str] = []
        for sid in ids:
            if sid not in seen:
                seen.add(sid)
                ordered.append(sid)
        return ordered

    def _on_copy(self) -> None:
        targets = self._selected_target_ids()
        if not targets:
            QMessageBox.information(
                self, "Copy decomposition", "Select one or more target spectra."
            )
            return

        overwrite: set[str] = set()
        to_copy: list[str] = []
        for target_id in targets:
            if self._controller.query.get_regions_ids(target_id):
                decision = self._ask_conflict(target_id)
                if decision == "cancel":
                    return
                if decision == "skip":
                    continue
                overwrite.add(target_id)
                to_copy.append(target_id)
            else:
                to_copy.append(target_id)

        if not to_copy:
            return

        link_flags = self._link_model.link_flags()
        applied = self._controller.copy_decomposition(
            self._source_spectrum_id,
            to_copy,
            link_flags,
            rescale_intensities=self._rescale_cb.isChecked(),
            overwrite_targets=overwrite,
            optimize_after=False,
        )
        if self._optimize_cb.isChecked() and applied:
            confirm_and_optimize(self, self._controller, spectrum_ids=applied)
        self.close()

    def _ask_conflict(self, spectrum_id: str) -> Literal["cancel", "skip", "overwrite"]:
        metadata = self._controller.query.get_metadata(spectrum_id)
        name = getattr(metadata, "name", None) or spectrum_id[:8]
        box = QMessageBox(self)
        box.setWindowTitle("Spectrum already fitted")
        box.setText(f"Spectrum “{name}” already has regions.")
        box.setInformativeText("Cancel returns to this dialog without changing your selection.")
        cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        skip_btn = box.addButton("Skip", QMessageBox.ButtonRole.ActionRole)
        overwrite_btn = box.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is skip_btn:
            return "skip"
        if clicked is overwrite_btn:
            return "overwrite"
        return "cancel"
