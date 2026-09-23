"""Popup constructor for parameter constraint expressions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from PySide6.QtCore import (
    QAbstractItemModel,
    QModelIndex,
    QPersistentModelIndex,
    QPoint,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QHideEvent,
    QIcon,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .assets import icon_path
from .component_colors import (
    STATUS_COLOR_EMPTY,
    STATUS_COLOR_PEAKS,
    STATUS_COLOR_REGIONS,
    color_for_component,
)
from .expr_tokens import shortest_unique_prefix
from .name_id_delegate import (
    ComponentColorRole,
    NameWithIdDelegate,
    ObjectIdPrefixRole,
    ObjectIdRole,
)
from .theme import make_translucent_popup
from .tree_style import EditorTreeView, apply_editor_tree_style

if TYPE_CHECKING:
    from .controller import ControllerWrapper

_DEFAULT_INDEX = QModelIndex()
_ID_DISPLAY_CHARS = 5
_COPY_ICON = QIcon(str(icon_path("copy.svg")))
_CHECK_ICON = QIcon(str(icon_path("check.svg")))
_SEARCH_ICON = QIcon(str(icon_path("search.svg")))
_COPY_FEEDBACK_MS = 1200

_STATUS_COLORS = {
    "empty": STATUS_COLOR_EMPTY,
    "regions": STATUS_COLOR_REGIONS,
    "peaks": STATUS_COLOR_PEAKS,
}
_STATUS_RANK = {"empty": 0, "regions": 1, "peaks": 2}

ExprNodeKind = Literal["file", "group", "spectrum", "region", "component"]


@dataclass
class ExprPickerItem:
    """Node in the expression component picker tree."""

    label: str
    kind: ExprNodeKind
    parent: ExprPickerItem | None = None
    object_id: str | None = None
    component_kind: Literal["peak", "background"] | None = None
    color_index: int = 0
    search_text: str = ""
    _row: int = 0
    children: list[ExprPickerItem] = field(default_factory=list)

    def child(self, row: int) -> ExprPickerItem | None:
        """Return the child at ``row``, if any."""
        if 0 <= row < len(self.children):
            return self.children[row]
        return None

    def row(self) -> int:
        """Return this item's index among its siblings."""
        return self._row

    def append_child(self, item: ExprPickerItem) -> None:
        """Append ``item`` and assign its sibling row."""
        item.parent = self
        item._row = len(self.children)
        self.children.append(item)


class ExprPickerModel(QAbstractItemModel):
    """
    File → group → spectrum → region → component tree for the expr popup.

    Leaves are peak/background components; only those expose an insertable id.
    """

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._root = ExprPickerItem(label="Root", kind="file")
        self._component_ids: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the hierarchy from the controller."""
        self.beginResetModel()
        self._root.children.clear()
        self._component_ids = []

        query = self._controller.query
        grouped: dict[Any, dict[Any, list[tuple[Any, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for spectrum_id in query.get_all_spectra_ids():
            metadata = query.get_metadata(spectrum_id)
            file_attr = getattr(metadata, "file", None)
            group_attr = getattr(metadata, "group", None)
            name_attr = getattr(metadata, "name", None)
            grouped[file_attr][group_attr].append((name_attr, spectrum_id))

        for file_key, groups in sorted(grouped.items(), key=lambda kv: str(kv[0] or "")):
            file_label = (str(file_key).split("/")[-1] if file_key else "No file") or "No file"
            file_item = ExprPickerItem(
                label=file_label,
                kind="file",
                search_text=file_label.lower(),
            )
            self._root.append_child(file_item)

            for group_key, spectra in sorted(groups.items(), key=lambda kv: str(kv[0] or "")):
                group_label = str(group_key) if group_key else "No group"
                group_item = ExprPickerItem(
                    label=group_label,
                    kind="group",
                    search_text=group_label.lower(),
                )
                file_item.append_child(group_item)

                for name, spectrum_id in sorted(spectra, key=lambda t: str(t[0] or "")):
                    spectrum_label = str(name) if name else "No name"
                    spectrum_item = ExprPickerItem(
                        label=spectrum_label,
                        kind="spectrum",
                        object_id=spectrum_id,
                        search_text=f"{spectrum_label} {spectrum_id}".lower(),
                    )
                    group_item.append_child(spectrum_item)

                    for region_index, region_id in enumerate(
                        query.get_regions_ids(spectrum_id), start=1
                    ):
                        region_label = f"Region {region_index}"
                        region_item = ExprPickerItem(
                            label=region_label,
                            kind="region",
                            object_id=region_id,
                            search_text=f"{region_label} {region_id}".lower(),
                        )
                        spectrum_item.append_child(region_item)

                        background_id = query.get_background_id(region_id)
                        if background_id is not None:
                            bg_dto = query.get_component_dto(background_id)
                            bg_label = bg_dto.name or "Background"
                            region_item.append_child(
                                ExprPickerItem(
                                    label=bg_label,
                                    kind="component",
                                    object_id=background_id,
                                    component_kind="background",
                                    search_text=f"{bg_label} {background_id}".lower(),
                                )
                            )
                            self._component_ids.append(background_id)

                        for peak_index, peak_id in enumerate(
                            query.get_peaks_ids(region_id), start=1
                        ):
                            peak_dto = query.get_component_dto(peak_id)
                            peak_label = peak_dto.name or f"Peak {peak_index}"
                            region_item.append_child(
                                ExprPickerItem(
                                    label=peak_label,
                                    kind="component",
                                    object_id=peak_id,
                                    component_kind="peak",
                                    color_index=peak_index - 1,
                                    search_text=f"{peak_label} {peak_id}".lower(),
                                )
                            )
                            self._component_ids.append(peak_id)

        self.endResetModel()

    @property
    def component_ids(self) -> list[str]:
        """Return all component ids present in the picker."""
        return list(self._component_ids)

    def insert_token_for(self, component_id: str) -> str:
        """Return the short unique prefix to insert for ``component_id``."""
        return shortest_unique_prefix(component_id, self._component_ids)

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX) -> int:
        """Return child count under ``parent``."""
        item = self._root if not parent.isValid() else parent.internalPointer()
        if not isinstance(item, ExprPickerItem):
            return 0
        return len(item.children)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX) -> int:
        """Return column count (always 1)."""
        del parent
        return 1

    def index(
        self,
        row: int,
        column: int,
        parent: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX,
    ) -> QModelIndex:
        """Return the child index at ``row`` under ``parent``."""
        if column != 0 or row < 0:
            return QModelIndex()
        parent_item = self._root if not parent.isValid() else parent.internalPointer()
        if not isinstance(parent_item, ExprPickerItem):
            return QModelIndex()
        child = parent_item.child(row)
        if child is None:
            return QModelIndex()
        return self.createIndex(row, column, child)

    def parent(self, index: QModelIndex | QPersistentModelIndex) -> QModelIndex:  # ty: ignore[invalid-method-override]
        """Return the parent of ``index``."""
        if not index.isValid():
            return QModelIndex()
        item = index.internalPointer()
        if not isinstance(item, ExprPickerItem):
            return QModelIndex()
        parent_item = item.parent
        if parent_item is None or parent_item is self._root:
            return QModelIndex()
        return self.createIndex(parent_item.row(), 0, parent_item)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """Return display label, id roles, and status/component colors."""
        if not index.isValid():
            return None
        item = index.internalPointer()
        if not isinstance(item, ExprPickerItem):
            return None

        if role == ObjectIdRole and item.object_id is not None:
            return item.object_id

        if role == ObjectIdPrefixRole and item.object_id is not None:
            if item.kind in {"spectrum", "region", "component"}:
                return item.object_id[:_ID_DISPLAY_CHARS]
            return None

        if role == ComponentColorRole:
            if item.kind == "component" and item.object_id is not None:
                return color_for_component(
                    kind=item.component_kind or "peak",
                    index=item.color_index,
                )
            if item.kind in {"spectrum", "file", "group"}:
                return _STATUS_COLORS.get(self._structure_status(item))
            return None

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return item.label

        return None

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        """Enable all rows; components are selectable for insertion."""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def item_from_index(self, index: QModelIndex) -> ExprPickerItem | None:
        """Return the backing item for ``index``."""
        if not index.isValid():
            return None
        item = index.internalPointer()
        return item if isinstance(item, ExprPickerItem) else None

    def find_index_for_object_id(self, object_id: str | None) -> QModelIndex | None:
        """Return the first index whose ``object_id`` matches."""
        if not object_id:
            return None

        def walk(parent: QModelIndex) -> QModelIndex | None:
            for row in range(self.rowCount(parent)):
                idx = self.index(row, 0, parent)
                item = self.item_from_index(idx)
                if item is not None and item.object_id == object_id:
                    return idx
                found = walk(idx)
                if found is not None:
                    return found
            return None

        return walk(QModelIndex())

    def _structure_status(self, item: ExprPickerItem) -> str:
        """Aggregate structure status like the spectrum tree."""
        query = self._controller.query
        if item.kind == "spectrum" and item.object_id is not None:
            return query.get_spectrum_structure_status(item.object_id)
        if item.kind in {"file", "group"}:
            best = "empty"
            for spectrum_id in self._spectrum_ids_under(item):
                status = query.get_spectrum_structure_status(spectrum_id)
                if _STATUS_RANK[status] > _STATUS_RANK[best]:
                    best = status
            return best
        return "empty"

    def _spectrum_ids_under(self, item: ExprPickerItem) -> list[str]:
        """Collect spectrum ids under ``item``."""
        ids: list[str] = []

        def walk(node: ExprPickerItem) -> None:
            if node.kind == "spectrum" and node.object_id is not None:
                ids.append(node.object_id)
            for child in node.children:
                walk(child)

        walk(item)
        return ids


class ExprEditorPopup(QFrame):
    """
    Overlay popup: expression field with copy, search, and component tree.

    Parameters
    ----------
    controller : ControllerWrapper
        Application controller for hierarchy data.
    initial_text : str
        Current expression text.
    focus_spectrum_id, focus_region_id : str or None, optional
        Expand/scroll target when the popup opens.
    parent : QWidget or None, optional
        Parent widget.
    """

    accepted = Signal(object)  # str
    cancelled = Signal()

    def __init__(
        self,
        controller: ControllerWrapper,
        *,
        initial_text: str = "",
        focus_spectrum_id: str | None = None,
        focus_region_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("ExprEditorPopup")
        make_translucent_popup(self)
        self._controller = controller
        self._committed = False
        self._copy_feedback_token = 0
        self._focus_spectrum_id = focus_spectrum_id
        self._focus_region_id = focus_region_id

        self._expr_edit = QLineEdit(self)
        self._expr_edit.setObjectName("ExprEditorField")
        self._expr_edit.setText(initial_text)
        self._expr_edit.setPlaceholderText("Expression…")

        self._copy_btn = QToolButton(self)
        self._copy_btn.setObjectName("ExprCopyButton")
        self._copy_btn.setIcon(_COPY_ICON)
        self._copy_btn.setIconSize(QSize(14, 14))
        self._copy_btn.setToolTip("Copy expression")
        self._copy_btn.clicked.connect(self._on_copy)

        self._search = QLineEdit(self)
        self._search.setObjectName("ExprSearchField")
        self._search.setPlaceholderText("Search file, group, spectrum, region, component…")
        self._search.addAction(_SEARCH_ICON, QLineEdit.ActionPosition.LeadingPosition)
        self._search.textChanged.connect(self._apply_filter)

        self._model = ExprPickerModel(controller, self)
        self._tree = EditorTreeView(self)
        self._tree.setModel(self._model)
        self._tree.setItemDelegate(NameWithIdDelegate(self._tree, swatch_size=6))
        self._tree.setHeaderHidden(True)
        self._tree.setSelectionMode(EditorTreeView.SelectionMode.SingleSelection)
        apply_editor_tree_style(self._tree, row_separators=False)
        self._tree.setMinimumHeight(220)
        self._tree.clicked.connect(self._on_tree_clicked)

        expr_row = QHBoxLayout()
        expr_row.setContentsMargins(0, 0, 0, 0)
        expr_row.setSpacing(4)
        expr_row.addWidget(self._expr_edit, stretch=1)
        expr_row.addWidget(self._copy_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addLayout(expr_row)
        layout.addWidget(self._search)
        layout.addWidget(self._tree)

        self._expr_edit.returnPressed.connect(self._commit)
        self.setMinimumWidth(420)
        self.resize(460, 380)

        QTimer.singleShot(0, self._finish_open)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint an opaque rounded panel so the popup separates from the app chrome."""
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, float(theme.POPUP_RADIUS), float(theme.POPUP_RADIUS))
        painter.fillPath(path, QColor(theme.SURFACE))
        painter.setPen(QPen(QColor(theme.POPUP_BORDER), 1.5))
        painter.drawPath(path)

    def _finish_open(self) -> None:
        """Focus the expression field and expand to the editing context."""
        self._expr_edit.setFocus(Qt.FocusReason.PopupFocusReason)
        self._expr_edit.setCursorPosition(len(self._expr_edit.text()))
        self._expand_to_focus()

    def _expand_to_focus(self) -> None:
        """Expand ancestors of the focus region/spectrum."""
        target_id = self._focus_region_id or self._focus_spectrum_id
        index = self._model.find_index_for_object_id(target_id)
        if index is None and self._focus_spectrum_id is not None:
            index = self._model.find_index_for_object_id(self._focus_spectrum_id)
        if index is None:
            self._tree.expandToDepth(1)
            return
        parent = index.parent()
        while parent.isValid():
            self._tree.expand(parent)
            parent = parent.parent()
        self._tree.expand(index)
        self._tree.scrollTo(index)
        self._tree.setCurrentIndex(index)

    def _apply_filter(self, text: str) -> None:
        """Show only rows matching ``text`` (and their ancestors)."""
        needle = text.strip().lower()
        if not needle:
            self._set_all_hidden(False)
            self._expand_to_focus()
            return

        visible_ids = self._collect_visible_item_ids(needle)
        self._apply_visibility(QModelIndex(), visible_ids)
        for item_id in visible_ids:
            # Expand non-leaf matches so filtered children stay reachable.
            index = self._index_for_item_id(item_id)
            if index is None:
                continue
            item = self._model.item_from_index(index)
            parent = index.parent()
            while parent.isValid():
                self._tree.expand(parent)
                parent = parent.parent()
            if item is not None and item.kind != "component":
                self._tree.expand(index)

    def _collect_visible_item_ids(self, needle: str) -> set[int]:
        """Return ``id(item)`` for nodes that match or have a matching descendant."""
        visible: set[int] = set()

        def walk(parent: QModelIndex) -> bool:
            any_hit = False
            for row in range(self._model.rowCount(parent)):
                idx = self._model.index(row, 0, parent)
                item = self._model.item_from_index(idx)
                if item is None:
                    continue
                child_hit = walk(idx)
                self_hit = needle in item.search_text
                if self_hit or child_hit:
                    visible.add(id(item))
                    any_hit = True
            return any_hit

        walk(QModelIndex())
        return visible

    def _index_for_item_id(self, item_id: int) -> QModelIndex | None:
        """Locate the model index for a Python ``id(item)``."""

        def walk(parent: QModelIndex) -> QModelIndex | None:
            for row in range(self._model.rowCount(parent)):
                idx = self._model.index(row, 0, parent)
                item = self._model.item_from_index(idx)
                if item is not None and id(item) == item_id:
                    return idx
                found = walk(idx)
                if found is not None:
                    return found
            return None

        return walk(QModelIndex())

    def _apply_visibility(self, parent: QModelIndex, visible_ids: set[int]) -> None:
        """Hide rows under ``parent`` that are not in ``visible_ids``."""
        for row in range(self._model.rowCount(parent)):
            idx = self._model.index(row, 0, parent)
            item = self._model.item_from_index(idx)
            hide = item is None or id(item) not in visible_ids
            self._tree.setRowHidden(row, parent, hide)
            self._apply_visibility(idx, visible_ids)

    def _set_all_hidden(self, hidden: bool) -> None:
        """Show or hide every row under the root."""

        def walk(parent: QModelIndex) -> None:
            for row in range(self._model.rowCount(parent)):
                idx = self._model.index(row, 0, parent)
                self._tree.setRowHidden(row, parent, hidden)
                walk(idx)

        walk(QModelIndex())

    def _on_tree_clicked(self, index: QModelIndex) -> None:
        """Insert a component id token when a component row is clicked."""
        item = self._model.item_from_index(index)
        if item is None or item.kind != "component" or item.object_id is None:
            return
        token = self._model.insert_token_for(item.object_id)
        cursor = self._expr_edit.cursorPosition()
        text = self._expr_edit.text()
        new_text = text[:cursor] + token + text[cursor:]
        self._expr_edit.setText(new_text)
        self._expr_edit.setCursorPosition(cursor + len(token))
        self._expr_edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_copy(self) -> None:
        """Copy the expression text and briefly show a check icon."""
        QApplication.clipboard().setText(self._expr_edit.text())
        self._copy_feedback_token += 1
        token = self._copy_feedback_token
        self._copy_btn.setIcon(_CHECK_ICON)

        def _restore() -> None:
            if token == self._copy_feedback_token:
                self._copy_btn.setIcon(_COPY_ICON)

        QTimer.singleShot(_COPY_FEEDBACK_MS, _restore)

    def _commit(self) -> None:
        """Emit the expression text and close."""
        if self._committed:
            return
        self._committed = True
        self.accepted.emit(self._expr_edit.text().strip())
        self.close()

    def _cancel(self) -> None:
        """Close without committing."""
        if self._committed:
            return
        self._committed = True
        self.cancelled.emit()
        self.close()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Escape cancels; Enter commits when the expression field is focused."""
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._expr_edit.hasFocus() or self._search.hasFocus():
                if self._search.hasFocus():
                    super().keyPressEvent(event)
                    return
                self._commit()
                event.accept()
                return
        super().keyPressEvent(event)

    def hideEvent(self, event: QHideEvent) -> None:
        """Commit when the popup is dismissed by clicking outside."""
        if not self._committed:
            self._commit()
        super().hideEvent(event)

    @staticmethod
    def open_for(
        controller: ControllerWrapper,
        *,
        initial_text: str,
        focus_spectrum_id: str | None = None,
        focus_region_id: str | None = None,
        parent: QWidget | None = None,
        on_accepted: Any = None,
    ) -> ExprEditorPopup:
        """
        Create, center on the application window, and show a popup.

        Parameters
        ----------
        controller : ControllerWrapper
            Application controller.
        initial_text : str
            Current expression.
        focus_spectrum_id, focus_region_id : str or None, optional
            Context to expand.
        parent : QWidget or None, optional
            Anchor widget; the popup is centered on its top-level window.
        on_accepted : callable or None, optional
            Called with ``str`` when the user commits.
        """
        popup = ExprEditorPopup(
            controller,
            initial_text=initial_text,
            focus_spectrum_id=focus_spectrum_id,
            focus_region_id=focus_region_id,
            parent=parent,
        )
        if on_accepted is not None:
            popup.accepted.connect(on_accepted)
        popup._center_on_app_window(parent)
        popup.show()
        return popup

    def _center_on_app_window(self, anchor: QWidget | None) -> None:
        """Move this popup to the center of the application main window."""
        window = anchor.window() if anchor is not None else None
        if window is None or window is self:
            screen = QApplication.primaryScreen()
            if screen is None:
                return
            geo = screen.availableGeometry()
        else:
            geo = window.frameGeometry()
        size = self.size()
        x = geo.x() + max(0, (geo.width() - size.width()) // 2)
        y = geo.y() + max(0, (geo.height() - size.height()) // 2)
        self.move(QPoint(x, y))
