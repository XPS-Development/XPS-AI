from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from PySide6.QtCore import QAbstractItemModel, QModelIndex, QPoint, Qt
from PySide6.QtWidgets import QComboBox, QMenu, QStyledItemDelegate, QTreeView, QWidget

from app.command.changes import ParameterField
from core.math_models import ModelRegistry

from .controller import ControllerWrapper


class ItemKind(Enum):
    """Kind of node in the properties tree."""

    ROOT = "root"
    REGION = "region"
    REGION_SLICE = "region_slice"
    COMPONENT = "component"
    COMPONENT_MODEL = "component_model"
    PARAMETER = "parameter"
    PARAMETER_FIELD = "parameter_field"


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

    Each item represents either a logical group (region, background, peak) or a
    single name/value row. Only leaf nodes carry a non-empty ``value`` field.

    Parameters
    ----------
    name : str
        Display name shown in the first column.
    value : Any, optional
        Raw value for the second column; display uses _format_value(value).
    parent : PropertyItem or None, optional
        Parent item in the tree.
    """

    name: str
    value: Any = None
    parent: Optional["PropertyItem"] = None
    children: list["PropertyItem"] = field(default_factory=list)
    kind: ItemKind = ItemKind.ROOT
    region_id: Optional[str] = None
    component_id: Optional[str] = None
    parameter_name: Optional[str] = None
    parameter_field: Optional[ParameterField] = None
    component_kind: Optional[Literal["peak", "background"]] = None

    def child(self, row: int) -> Optional["PropertyItem"]:
        """Return the child at the given row index."""
        if 0 <= row < len(self.children):
            return self.children[row]
        return None

    def row(self) -> int:
        """Return the index of this item within its parent."""
        return self.parent.children.index(self) if self.parent else 0

    def append_child(self, item: "PropertyItem") -> None:
        """Append a child item to this node."""
        self.children.append(item)


class PropertiesModel(QAbstractItemModel):
    """
    Read-only tree model for the Properties panel.

    The model presents regions of the currently selected spectrum along with
    their slices, background component, and peaks. All values are displayed as
    strings and cannot be edited.
    """

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._root_item = PropertyItem(name="Root", kind=ItemKind.ROOT)

    def _item(self, index: QModelIndex) -> Optional[PropertyItem]:
        """Return the item for the given index, or None if invalid."""
        return index.internalPointer() if index.isValid() else None

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
        return 2

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

        return self.createIndex(parent_item.row(), 0, parent_item)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        item = self._item(index)
        if not isinstance(item, PropertyItem):
            return None

        if role == Qt.CheckStateRole and index.column() == 1:
            if item.kind == ItemKind.PARAMETER_FIELD and item.parameter_field == "vary":
                if item.component_id and item.parameter_name:
                    dto = self._controller.query.get_component_dto(item.component_id)
                    param = dto.parameters[item.parameter_name]
                    return Qt.Checked if param.vary else Qt.Unchecked

        if role in (Qt.DisplayRole, Qt.EditRole):
            if index.column() == 0:
                return item.name
            if index.column() == 1:
                if item.kind == ItemKind.PARAMETER_FIELD and item.parameter_field == "vary":
                    return ""  # checkbox only, no text
                return _format_value(item.value)

        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:  # noqa: N802
        item = self._item(index)
        if not isinstance(item, PropertyItem):
            return Qt.NoItemFlags

        base_flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() != 1:
            return base_flags

        if item.kind == ItemKind.PARAMETER_FIELD and item.parameter_field == "vary":
            return base_flags | Qt.ItemIsUserCheckable
        if item.kind in {ItemKind.REGION_SLICE, ItemKind.PARAMETER_FIELD, ItemKind.COMPONENT_MODEL}:
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
        """Add parameter group and value/lower/upper/vary/expr rows for each parameter."""
        for param_name, param_dto in parameters_dto.items():
            param_group = PropertyItem(
                name=str(param_name),
                parent=parent_item,
                kind=ItemKind.PARAMETER,
                component_id=component_id,
                parameter_name=str(param_name),
            )
            parent_item.append_child(param_group)
            for field_name in ("value", "lower", "upper", "vary", "expr"):
                field_value = getattr(param_dto, field_name)
                param_group.append_child(
                    PropertyItem(
                        name=field_name,
                        value=field_value,
                        parent=param_group,
                        kind=ItemKind.PARAMETER_FIELD,
                        component_id=component_id,
                        parameter_name=str(param_name),
                        parameter_field=field_name,  # type: ignore[arg-type]
                    )
                )

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:  # noqa: N802
        item = self._item(index)
        if not isinstance(item, PropertyItem) or index.column() != 1:
            return False

        if role == Qt.CheckStateRole:
            if (
                item.kind == ItemKind.PARAMETER_FIELD
                and item.parameter_field == "vary"
                and item.component_id is not None
                and item.parameter_name is not None
            ):
                coerced = value == Qt.Checked.value
                try:
                    self._controller.update_parameter(
                        item.component_id,
                        item.parameter_name,
                        "vary",
                        coerced,
                        normalized=False,
                    )
                except Exception:  # noqa: BLE001
                    return False
                self.dataChanged.emit(index, index, [Qt.CheckStateRole])
                return True
            return False

        if role != Qt.EditRole:
            return False

        if item.kind == ItemKind.REGION_SLICE and item.region_id is not None:
            slice_mode = self._controller.get_app_parameters().region_slice_display_mode
            try:
                new_bound = int(value) if slice_mode == "index" else float(value)
            except (TypeError, ValueError):
                return False

            start, stop = self._controller.query.get_region_slice(item.region_id, mode=slice_mode)
            start = start if start is not None else (0 if slice_mode == "index" else 0.0)
            stop = stop if stop is not None else (0 if slice_mode == "index" else 0.0)

            if item.name == "start":
                start = new_bound
            elif item.name == "stop":
                stop = new_bound
            else:
                return False

            try:
                self._controller.update_region_slice(item.region_id, start, stop, mode=slice_mode)
            except Exception:  # noqa: BLE001
                return False

            item.value = new_bound
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True

        if (
            item.kind == ItemKind.COMPONENT_MODEL
            and item.component_id
            and item.region_id
            and item.component_kind
        ):
            new_model_name = str(value).strip()
            if not new_model_name:
                return False
            if new_model_name == item.value:
                return True  # no-op: model unchanged
            try:
                if item.component_kind == "peak":
                    self._controller.replace_peak_model(item.component_id, new_model_name)
                else:
                    self._controller.replace_background_model(item.region_id, new_model_name)
            except Exception:  # noqa: BLE001
                return False
            item.value = new_model_name
            self.refresh()
            return True

        if (
            item.kind == ItemKind.PARAMETER_FIELD
            and item.component_id is not None
            and item.parameter_name is not None
        ):
            field = item.parameter_field
            if field is None or field == "vary":
                return False  # vary is edited via checkbox only

            try:
                coerced: str | bool | float | None
                text = str(value)
                if field in {"value", "lower", "upper"}:
                    coerced = float(text)
                elif field == "expr":
                    coerced = text.strip() if text.strip() else None
                elif field == "name":
                    coerced = text
                else:
                    return False
            except (TypeError, ValueError):
                return False

            try:
                self._controller.update_parameter(
                    item.component_id,
                    item.parameter_name,
                    field,
                    coerced,
                    normalized=False,
                )
            except Exception:  # noqa: BLE001
                return False

            item.value = coerced
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
        if item.kind == ItemKind.PARAMETER_FIELD and item.parameter_field == "vary":
            return None
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

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._model = PropertiesModel(controller, self)
        self.setModel(self._model)
        self.setItemDelegateForColumn(1, PropertiesDelegate(self))
        self.setIndentation(12)
        self.setHeaderHidden(False)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_custom_context_menu)
        self.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.refresh()

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
        Refresh tree contents from the controller and expand nodes.
        """
        self._model.refresh()
        self.expandAll()

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

    def _on_custom_context_menu(self, pos: QPoint) -> None:
        """
        Show a context menu for creating regions, peaks, and background.

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

        # Resolve component from clicked item (e.g. Background or Peak node).
        component_item = item
        while component_item is not None and component_item.kind != ItemKind.COMPONENT:
            component_item = component_item.parent
        component_id = component_item.component_id if component_item else None

        menu = QMenu(self)

        if spectrum_id is not None:
            menu.addAction("Add region", self._create_full_region_for_spectrum)

        if region_id is not None:
            menu.addAction("Add peak", lambda rid=region_id: self._create_peak_for_region(rid))
            menu.addAction("Set background", lambda rid=region_id: self._set_background_for_region(rid))

        if region_id is not None and item is not None:
            menu.addAction("Delete region", lambda rid=region_id: self._delete_region(rid))

        if component_id is not None:
            menu.addAction("Delete component", lambda cid=component_id: self._delete_component(cid))

        if not menu.actions():
            return

        menu.exec(self.viewport().mapToGlobal(pos))

    def _create_full_region_for_spectrum(self) -> None:
        """
        Create a new region covering the full spectrum for the current selection.
        """
        spectrum_id = self._controller.selected_spectrum_id
        if spectrum_id is None:
            return

        try:
            spectrum_dto = self._controller.query.get_spectrum_dto(spectrum_id, normalized=False)
        except Exception:  # noqa: BLE001
            return

        length = len(spectrum_dto.x)
        if length <= 1:
            return

        try:
            self._controller.create_region(spectrum_id, 0, length)
        except Exception:  # noqa: BLE001
            return

    def _create_peak_for_region(self, region_id: str) -> None:
        """
        Create a new peak with default model for the given region.

        Parameters
        ----------
        region_id : str
            Identifier of the parent region.
        """
        try:
            self._controller.create_peak(region_id, "pseudo-voigt", parameters=None)
        except Exception:  # noqa: BLE001
            return

    def _set_background_for_region(self, region_id: str) -> None:
        """
        Create or replace the background with a default model for the given region.

        Parameters
        ----------
        region_id : str
            Identifier of the parent region.
        """
        try:
            self._controller.create_background(region_id, "shirley", parameters=None)
        except Exception:  # noqa: BLE001
            return

    def _delete_region(self, region_id: str) -> None:
        """
        Remove the region and refresh the properties view; clear selection if needed.

        Parameters
        ----------
        region_id : str
            Identifier of the region to remove.
        """
        try:
            if self._controller.selected_region_id == region_id:
                self._controller.set_selection(self._controller.selected_spectrum_id, None)
            self._controller.full_remove_object(region_id)
            self.refresh()
        except Exception:  # noqa: BLE001
            return

    def _delete_component(self, component_id: str) -> None:
        """
        Remove the component (peak or background) and refresh the properties view.

        Parameters
        ----------
        component_id : str
            Identifier of the component to remove.
        """
        try:
            self._controller.full_remove_object(component_id)
            self.refresh()
        except Exception:  # noqa: BLE001
            return
