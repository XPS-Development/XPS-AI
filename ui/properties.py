from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from PySide6.QtCore import QAbstractItemModel, QModelIndex, QPoint, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHeaderView,
    QMenu,
    QStyledItemDelegate,
    QTreeView,
    QWidget,
)

from core.math_models import ModelRegistry

from .context_menus import attach_region_context_actions, attach_spectrum_context_actions
from .controller import ControllerWrapper


class ItemKind(Enum):
    """Kind of node in the properties tree."""

    ROOT = "root"
    REGION = "region"
    REGION_SLICE = "region_slice"
    COMPONENT = "component"
    COMPONENT_MODEL = "component_model"
    PARAMETER_ROW = "parameter_row"


def _format_value(val: Any) -> str:
    """
    Format a value for display in the properties tree.

    None -> \"\", bool as-is, numbers with 2 decimal places, else str(val).
    """
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (int, float)):
        return f"{float(val):.2f}"
    return str(val)


@dataclass(eq=False)
class PropertyItem:
    """
    Node used by :class:`PropertiesModel`.

    Each item represents either a logical group (region, background, peak), a
    slice bound, a model selector row, or one parameter row (value/lower/upper
    /vary/expr across columns 1–5).

    Parameters
    ----------
    name : str
        Display name shown in the first column.
    value : Any, optional
        For ``REGION_SLICE`` / ``COMPONENT_MODEL``, the bound or model name.
        For ``PARAMETER_ROW``, the parameter's value (column 1).
    parent : PropertyItem or None, optional
        Parent item in the tree.
    param_lower, param_upper, param_vary, param_expr : optional
        Used when ``kind`` is ``PARAMETER_ROW`` (columns 2–5).
    """

    name: str
    value: Any = None
    parent: Optional["PropertyItem"] = None
    children: list["PropertyItem"] = field(default_factory=list)
    kind: ItemKind = ItemKind.ROOT
    region_id: Optional[str] = None
    component_id: Optional[str] = None
    parameter_name: Optional[str] = None
    component_kind: Optional[Literal["peak", "background"]] = None
    param_lower: Any = None
    param_upper: Any = None
    param_vary: bool = False
    param_expr: Any = None

    def child(self, row: int) -> Optional["PropertyItem"]:
        """Return the child at the given row index."""
        if 0 <= row < len(self.children):
            return self.children[row]
        return None

    def row(self) -> int:
        """Return the index of this item within its parent."""
        if self.parent is None:
            return 0
        try:
            return self.parent.children.index(self)
        except ValueError:
            # This can happen if Qt holds a stale QModelIndex whose internalPointer()
            # refers to an item that was removed during a model reset/rebuild.
            return -1

    def append_child(self, item: "PropertyItem") -> None:
        """Append a child item to this node."""
        self.children.append(item)


class PropertiesModel(QAbstractItemModel):
    """
    Tree-table model for the Properties panel (six columns).

    Presents regions of the selected spectrum with slice bounds, components,
    model selection, and one row per fit parameter (value, lower, upper, vary,
    expr). Slice, model, and parameter cells are editable where applicable.
    """

    _HEADER_LABELS = ("Name", "Value", "Lower", "Upper", "Vary", "Expr")

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._root_item = PropertyItem(name="Root", kind=ItemKind.ROOT)

    def _item(self, index: QModelIndex) -> Optional[PropertyItem]:
        """Return the item for the given index, or None if invalid."""
        if not index.isValid():
            return None
        ptr = index.internalPointer()
        return ptr if isinstance(ptr, PropertyItem) else None

    # ------------------------------------------------------------------
    # Required model API
    # ------------------------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if not parent.isValid():
            item = self._root_item
        else:
            item = self._item(parent)
        if not isinstance(item, PropertyItem):
            return 0
        return len(item.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        del parent
        return len(self._HEADER_LABELS)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.DisplayRole,
    ) -> Any:
        if role != Qt.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        if 0 <= section < len(self._HEADER_LABELS):
            return self._HEADER_LABELS[section]
        return None

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:  # noqa: N802
        if row < 0 or column < 0:
            return QModelIndex()

        parent_item = self._root_item if not parent.isValid() else self._item(parent)
        if not isinstance(parent_item, PropertyItem):
            return QModelIndex()

        child_item = parent_item.child(row)
        if child_item is None:
            return QModelIndex()
        return self.createIndex(row, column, child_item)

    def parent(self, index: QModelIndex) -> QModelIndex:  # noqa: N802
        item = self._item(index)
        if not isinstance(item, PropertyItem):
            return QModelIndex()

        parent_item = item.parent
        if parent_item is None or parent_item is self._root_item:
            return QModelIndex()

        row = parent_item.row()
        if row < 0:
            return QModelIndex()
        return self.createIndex(row, 0, parent_item)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        item = self._item(index)
        if not isinstance(item, PropertyItem):
            return None

        col = index.column()

        if role == Qt.CheckStateRole and col == 4 and item.kind == ItemKind.PARAMETER_ROW:
            return Qt.Checked if item.param_vary else Qt.Unchecked

        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == 0:
                return item.name
            if item.kind == ItemKind.PARAMETER_ROW:
                if col == 1:
                    return _format_value(item.value)
                if col == 2:
                    return _format_value(item.param_lower)
                if col == 3:
                    return _format_value(item.param_upper)
                if col == 4:
                    return ""
                if col == 5:
                    return "" if item.param_expr is None else str(item.param_expr)
            if item.kind == ItemKind.REGION_SLICE and col == 1:
                return _format_value(item.value)
            if item.kind == ItemKind.COMPONENT_MODEL and col == 1:
                return _format_value(item.value)

        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:  # noqa: N802
        item = self._item(index)
        if not isinstance(item, PropertyItem):
            return Qt.NoItemFlags

        base_flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        col = index.column()

        if item.kind == ItemKind.PARAMETER_ROW:
            if col == 4:
                return base_flags | Qt.ItemIsUserCheckable
            if col in (1, 2, 3, 5):
                return base_flags | Qt.ItemIsEditable
            return base_flags

        if col == 1 and item.kind in {ItemKind.REGION_SLICE, ItemKind.COMPONENT_MODEL}:
            return base_flags | Qt.ItemIsEditable

        return base_flags

    def _add_region_slice(
        self,
        parent_item: PropertyItem,
        region_id: str,
        start_val: int | float,
        stop_val: int | float,
    ) -> None:
        """Add start and stop rows under parent for the given region slice."""
        parent_item.append_child(
            PropertyItem(
                name="start",
                value=start_val,
                parent=parent_item,
                kind=ItemKind.REGION_SLICE,
                region_id=region_id,
            )
        )
        parent_item.append_child(
            PropertyItem(
                name="stop",
                value=stop_val,
                parent=parent_item,
                kind=ItemKind.REGION_SLICE,
                region_id=region_id,
            )
        )

    def _add_parameters(
        self,
        parent_item: PropertyItem,
        component_id: str,
        parameters_dto: dict[str, Any],
    ) -> None:
        """Add one flat row per parameter (value, lower, upper, vary, expr)."""
        for param_name, param_dto in parameters_dto.items():
            parent_item.append_child(
                PropertyItem(
                    name=str(param_name),
                    value=param_dto.value,
                    parent=parent_item,
                    kind=ItemKind.PARAMETER_ROW,
                    component_id=component_id,
                    parameter_name=str(param_name),
                    param_lower=param_dto.lower,
                    param_upper=param_dto.upper,
                    param_vary=bool(param_dto.vary),
                    param_expr=param_dto.expr,
                )
            )

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:  # noqa: N802
        item = self._item(index)
        if not isinstance(item, PropertyItem):
            return False

        col = index.column()

        if role == Qt.CheckStateRole:
            if (
                item.kind == ItemKind.PARAMETER_ROW
                and col == 4
                and item.component_id is not None
                and item.parameter_name is not None
            ):
                coerced = value == Qt.Checked.value
                self._controller.update_parameter(
                    item.component_id,
                    item.parameter_name,
                    "vary",
                    coerced,
                    normalized=False,
                )
                item.param_vary = coerced
                self.dataChanged.emit(index, index, [Qt.CheckStateRole])
                return True
            return False

        if role != Qt.EditRole:
            return False

        if item.kind == ItemKind.REGION_SLICE and col == 1 and item.region_id is not None:
            slice_mode = self._controller.get_app_parameters().region_slice_display_mode
            new_bound = int(value) if slice_mode == "index" else float(value)

            start, stop = self._controller.query.get_region_slice(item.region_id, mode=slice_mode)
            start = start if start is not None else (0 if slice_mode == "index" else 0.0)
            stop = stop if stop is not None else (0 if slice_mode == "index" else 0.0)

            if item.name == "start":
                start = new_bound
            elif item.name == "stop":
                stop = new_bound
            else:
                return False

            self._controller.update_region_slice(item.region_id, start, stop, mode=slice_mode)

            item.value = new_bound
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True

        if (
            item.kind == ItemKind.COMPONENT_MODEL
            and col == 1
            and item.component_id
            and item.region_id
            and item.component_kind
        ):
            new_model_name = str(value).strip()
            if not new_model_name:
                return False
            if new_model_name == item.value:
                return True  # no-op: model unchanged
            if item.component_kind == "peak":
                self._controller.replace_peak_model(item.component_id, new_model_name)
            else:
                self._controller.replace_background_model(item.region_id, new_model_name)
            item.value = new_model_name
            self.refresh()
            return True

        if (
            item.kind == ItemKind.PARAMETER_ROW
            and item.component_id is not None
            and item.parameter_name is not None
        ):
            field_by_col = {1: "value", 2: "lower", 3: "upper", 5: "expr"}
            field = field_by_col.get(col)
            if field is None:
                return False

            coerced: str | bool | float | None
            text = str(value)
            if field in {"value", "lower", "upper"}:
                coerced = float(text)
            elif field == "expr":
                coerced = text.strip() if text.strip() else None
            else:
                return False
            self._controller.update_parameter(
                item.component_id,
                item.parameter_name,
                field,  # type: ignore[arg-type]
                coerced,
                normalized=False,
            )
            if field == "value":
                item.value = coerced
            elif field == "lower":
                item.param_lower = coerced
            elif field == "upper":
                item.param_upper = coerced
            else:
                item.param_expr = coerced
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """
        Rebuild the tree from the controller's current selection.

        The model inspects the currently selected spectrum and its regions
        using the controller's context and DTO service. All values are turned
        into display strings.
        """
        self.beginResetModel()
        self._root_item.children.clear()

        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            self._root_item.append_child(PropertyItem(name="No spectrum selected", parent=self._root_item))
            self.endResetModel()
            return

        query = self._controller.query
        params = self._controller.get_app_parameters()
        slice_mode = params.region_slice_display_mode
        show_id = params.show_id_in_properties_tree

        region_ids = query.get_regions_ids(spectrum_id)
        if not region_ids:
            self._root_item.append_child(PropertyItem(name="No regions", parent=self._root_item))
            self.endResetModel()
            return

        for idx, region_id in enumerate(region_ids, start=1):
            region_label = f"Region {idx} {region_id[:5]}" if show_id else f"Region {idx}"
            region_item = PropertyItem(
                name=region_label,
                parent=self._root_item,
                kind=ItemKind.REGION,
                region_id=region_id,
            )
            self._root_item.append_child(region_item)

            start_val, stop_val = query.get_region_slice(region_id, mode=slice_mode)
            start_val = start_val if start_val is not None else (0 if slice_mode == "index" else 0.0)
            stop_val = stop_val if stop_val is not None else (0 if slice_mode == "index" else 0.0)
            self._add_region_slice(region_item, region_id, start_val, stop_val)

            background_id = query.get_background_id(region_id)
            peaks_ids = list(query.get_peaks_ids(region_id))

            if background_id is not None:
                background_name = f"Background {background_id[:5]}" if show_id else "Background"
                background_item = PropertyItem(
                    name=background_name,
                    parent=region_item,
                    kind=ItemKind.COMPONENT,
                    region_id=region_id,
                    component_id=background_id,
                )
                region_item.append_child(background_item)
                background_dto = query.get_component_dto(background_id)
                background_item.append_child(
                    PropertyItem(
                        name="model",
                        value=background_dto.model.name,
                        parent=background_item,
                        kind=ItemKind.COMPONENT_MODEL,
                        component_id=background_id,
                        region_id=region_id,
                        component_kind="background",
                    )
                )
                self._add_parameters(background_item, background_id, background_dto.parameters)

            for peak_index, peak_id in enumerate(peaks_ids, start=1):
                peak_name = f"Peak {peak_index} {peak_id[:5]}" if show_id else f"Peak {peak_index}"
                peak_item = PropertyItem(
                    name=peak_name,
                    parent=region_item,
                    kind=ItemKind.COMPONENT,
                    region_id=region_id,
                    component_id=peak_id,
                )
                region_item.append_child(peak_item)
                peak_dto = query.get_component_dto(peak_id)
                peak_item.append_child(
                    PropertyItem(
                        name="model",
                        value=peak_dto.model.name,
                        parent=peak_item,
                        kind=ItemKind.COMPONENT_MODEL,
                        component_id=peak_id,
                        region_id=region_id,
                        component_kind="peak",
                    )
                )
                self._add_parameters(peak_item, peak_id, peak_dto.parameters)

        self.endResetModel()


class PropertiesDelegate(QStyledItemDelegate):
    """
    Delegate that provides a combo box for the component model row in the value column.
    """

    def createEditor(
        self,
        parent: QWidget,
        option: Any,
        index: QModelIndex,
    ) -> QWidget:
        if index.column() != 1:
            return super().createEditor(parent, option, index)
        item = index.internalPointer() if index.isValid() else None
        if not isinstance(item, PropertyItem):
            return super().createEditor(parent, option, index)
        if item.kind != ItemKind.COMPONENT_MODEL:
            return super().createEditor(parent, option, index)
        combo = QComboBox(parent)
        if item.component_kind == "peak":
            combo.addItems(ModelRegistry.get_peak_model_names())
        else:
            combo.addItems(ModelRegistry.get_background_model_names())
        return combo

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        if index.column() != 1:
            super().setEditorData(editor, index)
            return
        item = index.internalPointer() if index.isValid() else None
        if not isinstance(item, PropertyItem) or item.kind != ItemKind.COMPONENT_MODEL:
            super().setEditorData(editor, index)
            return
        combo = editor
        if isinstance(combo, QComboBox):
            display = _format_value(item.value)
            idx = combo.findText(display)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def setModelData(
        self,
        editor: QWidget,
        model: QAbstractItemModel,
        index: QModelIndex,
    ) -> None:
        if index.column() != 1:
            super().setModelData(editor, model, index)
            return
        item = index.internalPointer() if index.isValid() else None
        if not isinstance(item, PropertyItem) or item.kind != ItemKind.COMPONENT_MODEL:
            super().setModelData(editor, model, index)
            return
        combo = editor
        if isinstance(combo, QComboBox):
            model.setData(index, combo.currentText(), Qt.EditRole)


class PropertiesView(QTreeView):
    """
    View used for the read-only Properties panel.

    The view owns a :class:`PropertiesModel` instance and exposes a convenience
    :meth:`refresh` method that can be connected directly to controller
    signals.
    """

    # Approximate character counts per column for initial pixel widths (see _apply_column_widths).
    _COLUMN_CHAR_WIDTHS = (11, 5, 4, 4, 3, 12)

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._model = PropertiesModel(controller, self)
        self.setModel(self._model)
        self.setItemDelegateForColumn(1, PropertiesDelegate(self))
        self.setIndentation(12)
        self.setHeaderHidden(False)
        hdr = self.header()
        hdr.setStretchLastSection(False)
        self._apply_column_widths()
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_custom_context_menu)
        self.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.refresh()

    def _apply_column_widths(self) -> None:
        """
        Set fixed header section widths from the view font and target character counts.

        Uses representative glyphs per column so Name/Expr allow letters and
        numeric columns use digit width; Vary uses a short digit span (~checkbox).
        """
        fm = QFontMetrics(self.font())
        hdr = self.header()
        padding = 14
        ref_chars = ("M", "0", "0", "0", "0", "m")
        for i, n in enumerate(self._COLUMN_CHAR_WIDTHS):
            ch = ref_chars[i]
            # hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
            hdr.resizeSection(i, fm.horizontalAdvance(ch * n) + padding)

    def model(self) -> PropertiesModel:  # type: ignore[override]
        """
        Return the underlying :class:`PropertiesModel` instance.

        Returns
        -------
        PropertiesModel
            Backing model used by this view.
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
    def _stable_key_for_item(item: PropertyItem) -> tuple[Any, ...]:
        """Return a key stable across model rebuilds for the given property item."""
        if item.kind == ItemKind.REGION:
            return ("region", item.region_id)
        if item.kind == ItemKind.COMPONENT:
            return ("component", item.region_id, item.component_id)
        if item.kind == ItemKind.COMPONENT_MODEL:
            return ("model", item.region_id, item.component_id)
        if item.kind == ItemKind.REGION_SLICE:
            return ("slice", item.region_id, item.name)
        if item.kind == ItemKind.PARAMETER_ROW:
            return ("param", item.component_id, item.parameter_name)
        return ("other", item.kind, item.name)

    def _collect_expanded_stable_keys(self) -> set[tuple[Any, ...]]:
        """Return stable keys for all expanded property rows."""
        keys: set[tuple[Any, ...]] = set()

        def walk(parent: QModelIndex) -> None:
            for row in range(self._model.rowCount(parent)):
                idx = self._model.index(row, 0, parent)
                if self.isExpanded(idx):
                    raw = idx.internalPointer()
                    if isinstance(raw, PropertyItem):
                        keys.add(self._stable_key_for_item(raw))
                walk(idx)

        walk(QModelIndex())
        return keys

    def _restore_expanded_stable_keys(self, keys: set[tuple[Any, ...]]) -> None:
        """Expand rows whose stable keys match a previously expanded set."""

        def walk(parent: QModelIndex) -> None:
            for row in range(self._model.rowCount(parent)):
                idx = self._model.index(row, 0, parent)
                raw = idx.internalPointer()
                if isinstance(raw, PropertyItem) and self._stable_key_for_item(raw) in keys:
                    self.setExpanded(idx, True)
                walk(idx)

        walk(QModelIndex())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selection_changed(self, selected: Any, _deselected: Any) -> None:
        """
        Update controller region selection from the properties view.

        Parameters
        ----------
        selected : Any
            Newly selected indexes.
        _deselected : Any
            No longer selected indexes (unused).
        """
        del _deselected

        indexes = selected.indexes()
        if not indexes:
            self._controller.set_selection(self._controller.selected_spectrum_id, None)
            return

        index = indexes[0]
        item = index.internalPointer()
        if not isinstance(item, PropertyItem):
            return

        region_item = item
        while region_item is not None and region_item.region_id is None:
            region_item = region_item.parent

        region_id = region_item.region_id if region_item is not None else None
        self._controller.set_selection(self._controller.selected_spectrum_id, region_id)

    def _selected_component_id(self) -> str | None:
        """Return the component id for the current selection, if any."""
        sel = self.selectionModel().selectedIndexes()
        idx = sel[0] if sel else self.selectionModel().currentIndex()
        if not idx.isValid():
            return None
        raw = idx.internalPointer()
        item = raw if isinstance(raw, PropertyItem) else None
        while item is not None and item.kind != ItemKind.COMPONENT:
            item = item.parent
        return item.component_id if item is not None else None

    def _copy_selected_component_id(self) -> None:
        """Copy the selected component id to the system clipboard."""
        cid = self._selected_component_id()
        if cid:
            QApplication.clipboard().setText(cid)

    def _delete_selected_component(self) -> None:
        """Remove the selected component and refresh."""
        cid = self._selected_component_id()
        if cid:
            self._controller.full_remove_object(cid)
            self.refresh()

    def _on_custom_context_menu(self, pos: QPoint) -> None:
        """
        Spectrum-level menu on empty space; region-level menu on rows with a region.

        Parameters
        ----------
        pos : QPoint
            Position of the context menu request.
        """
        index = self.indexAt(pos)
        item: PropertyItem | None
        if index.isValid():
            raw_item = index.internalPointer()
            item = raw_item if isinstance(raw_item, PropertyItem) else None
        else:
            item = None

        region_item = item
        while region_item is not None and region_item.region_id is None:
            region_item = region_item.parent

        region_id = region_item.region_id if region_item is not None else None
        spectrum_id = self._controller.selected_spectrum_id

        menu = QMenu(self)

        if region_id is not None:
            region_actions = attach_region_context_actions(menu, self._controller, region_id, self)
            region_actions.update_enabled_state()
            menu.addSeparator()
            sel_cid = self._selected_component_id()
            copy_action = menu.addAction("Copy ID", self._copy_selected_component_id)
            copy_action.setEnabled(sel_cid is not None)
            del_action = menu.addAction("Delete component", self._delete_selected_component)
            del_action.setEnabled(sel_cid is not None)
        elif spectrum_id is not None:
            spec_actions = attach_spectrum_context_actions(menu, self._controller, self)
            spec_actions.update_enabled_state()
        else:
            return

        menu.exec(self.viewport().mapToGlobal(pos))
