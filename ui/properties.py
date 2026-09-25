"""Properties tree for region slices, models, and component parameters."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

import numpy as np
from PySide6.QtCore import (
    QAbstractItemModel,
    QEvent,
    QItemSelectionModel,
    QModelIndex,
    QPersistentModelIndex,
    QPoint,
    QRect,
    Qt,
    QTimer,
)
from PySide6.QtGui import QColor, QIcon, QMouseEvent, QPainter, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QLineEdit,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeView,
    QWidget,
)

from core.math_models import ModelRegistry
from core.math_models.soft_ranges import soft_region_bound_range

from . import theme
from .assets import icon_path
from .component_colors import ID_SUFFIX_HEX, color_for_component
from .context_menus import attach_region_context_actions, attach_spectrum_context_actions
from .controller import ControllerWrapper
from .expr_editor_popup import ExprEditorPopup
from .name_id_delegate import (
    ComponentColorRole,
    NameWithIdDelegate,
    ObjectIdPrefixRole,
    ObjectIdRole,
)
from .optimize_confirm import confirm_and_optimize
from .parameter_value_editor import ParameterValueEditor
from .tree_style import EditorTreeView, apply_editor_tree_style

_DEFAULT_INDEX = QModelIndex()
_ID_DISPLAY_CHARS = 5


def _as_model_index(index: QModelIndex | QPersistentModelIndex) -> QModelIndex:
    """Return a plain ``QModelIndex`` (PySide stubs omit persistent conversion)."""
    model = index.model()
    if model is None:
        return QModelIndex()
    return model.index(index.row(), index.column(), index.parent())


class ItemKind(Enum):
    """Kind of node in the properties tree."""

    ROOT = "root"
    REGION = "region"
    REGION_SLICE = "region_slice"
    COMPONENT = "component"
    COMPONENT_MODEL = "component_model"
    PARAMETER_ROW = "parameter_row"
    PARAMETER_FIELD = "parameter_field"
    ACTION_ROW = "action_row"


ActionKind = Literal["add_region", "add_background", "add_peak"]


def _format_value(val: Any) -> str:
    r"""
    Format a value for display in the properties tree.

    None -> \"\", bool as-is, non-finite floats as \"—\", numbers with 2
    decimal places, else str(val).
    """
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (int, float)):
        number = float(val)
        if not math.isfinite(number):
            return "—"
        return f"{number:.2f}"
    return str(val)


@dataclass(eq=False)
class PropertyItem:
    """
    Node used by :class:`PropertiesModel`.

    Each item represents either a logical group (region, background, peak), a
    slice bound, a model selector row, a parameter summary row, or a nested
    constraint field (lower / upper / expr) under a parameter.

    Parameters
    ----------
    name : str
        Display name shown in the first column (without id suffix).
    value : Any, optional
        For ``REGION`` and peak ``COMPONENT`` rows, the read-only fitted area
        (column 1). For ``REGION_SLICE`` / ``COMPONENT_MODEL``, the bound or
        model name. For ``PARAMETER_ROW``, the parameter's value (column 1).
    parent : PropertyItem or None, optional
        Parent item in the tree.
    param_lower, param_upper, param_vary, param_expr : optional
        Used when ``kind`` is ``PARAMETER_ROW``; mirrored on nested
        ``PARAMETER_FIELD`` children (lower / upper / vary / expr).
    parameter_field : {"lower", "upper", "expr", "vary"} or None, optional
        Which constraint a ``PARAMETER_FIELD`` row edits.
    action : ActionKind or None, optional
        Which mutation an ``ACTION_ROW`` triggers.
    object_id : str or None, optional
        Full region/component id for gray suffix / clipboard when shown.
    stored_name : str or None, optional
        Optional user label for ``COMPONENT`` rows (edit buffer; may differ
        from the positional ``name`` fallback shown when unset).
    color_index : int, optional
        Zero-based peak order used to pick a palette color for peak swatches.
    """

    name: str
    value: Any = None
    parent: Optional["PropertyItem"] = None
    children: list["PropertyItem"] = field(default_factory=list)
    kind: ItemKind = ItemKind.ROOT
    region_id: str | None = None
    component_id: str | None = None
    parameter_name: str | None = None
    component_kind: Literal["peak", "background"] | None = None
    object_id: str | None = None
    stored_name: str | None = None
    soft_lo: float | None = None
    soft_hi: float | None = None
    param_lower: Any = None
    param_upper: Any = None
    param_vary: bool = False
    param_expr: Any = None
    parameter_field: Literal["lower", "upper", "expr", "vary"] | None = None
    action: ActionKind | None = None
    color_index: int = 0

    def child(self, row: int) -> Optional["PropertyItem"]:
        """Return the child at the given row index."""
        if 0 <= row < len(self.children):
            return self.children[row]
        return None

    def row(self) -> int:
        """Return the index of this item within its parent."""
        if self.parent is None:
            return 0
        for idx, child in enumerate(self.parent.children):
            if child is self:
                return idx
        # Qt can hold a stale QModelIndex whose internalPointer() refers to an
        # item that was removed during a model reset/rebuild.
        return -1

    def append_child(self, item: "PropertyItem") -> None:
        """Append a child item to this node."""
        self.children.append(item)


def _parameter_constraint_marks(item: PropertyItem) -> str:
    """Return a compact indicator glyph when an expression constraint is set."""
    if item.param_expr:
        return "ƒ"
    return ""


def _ancestor_region_id(item: PropertyItem) -> str | None:
    """Walk parents to find the nearest ``region_id``."""
    node: PropertyItem | None = item
    while node is not None:
        if node.region_id is not None:
            return node.region_id
        node = node.parent
    return None


class PropertiesModel(QAbstractItemModel):
    """
    Tree-table model for the Properties panel (two columns).

    Presents regions of the selected spectrum with slice bounds, components,
    model selection, and one row per fit parameter (value). Lower, upper, vary,
    and expr live as nested child rows under each parameter.
    """

    _HEADER_LABELS = ("Name", "Value")

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._root_item = PropertyItem(name="Root", kind=ItemKind.ROOT)

    def _item(self, index: QModelIndex | QPersistentModelIndex) -> PropertyItem | None:
        """Return the item for the given index, or None if invalid."""
        if not index.isValid():
            return None
        ptr = index.internalPointer()
        return ptr if isinstance(ptr, PropertyItem) else None

    # ------------------------------------------------------------------
    # Required model API
    # ------------------------------------------------------------------

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX) -> int:
        """Return the number of child rows under ``parent``."""
        if not parent.isValid():
            item = self._root_item
        else:
            item = self._item(parent)
        if not isinstance(item, PropertyItem):
            return 0
        return len(item.children)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX) -> int:
        """Return the number of columns in the model."""
        del parent
        return len(self._HEADER_LABELS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """Return header labels for horizontal display roles."""
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        if 0 <= section < len(self._HEADER_LABELS):
            return self._HEADER_LABELS[section]
        return None

    def index(
        self,
        row: int,
        column: int,
        parent: QModelIndex | QPersistentModelIndex = _DEFAULT_INDEX,
    ) -> QModelIndex:
        """Return the index of the child at ``row``, ``column`` under ``parent``."""
        if row < 0 or column < 0:
            return QModelIndex()

        parent_item = self._root_item if not parent.isValid() else self._item(parent)
        if not isinstance(parent_item, PropertyItem):
            return QModelIndex()

        child_item = parent_item.child(row)
        if child_item is None:
            return QModelIndex()
        return self.createIndex(row, column, child_item)

    def parent(self, index: QModelIndex | QPersistentModelIndex) -> QModelIndex:  # ty: ignore[invalid-method-override]
        """Return the parent index of ``index``."""
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

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """Return display, edit, decoration, or check-state data for ``index``."""
        item = self._item(index)
        if not isinstance(item, PropertyItem):
            return None

        col = index.column()

        if role == ObjectIdRole and col == 0 and item.object_id is not None:
            return item.object_id

        if role == ObjectIdPrefixRole and col == 0 and item.object_id is not None:
            if self._controller.get_app_parameters().show_id_in_properties_tree:
                return item.object_id[:_ID_DISPLAY_CHARS]
            return None

        if (
            role == ComponentColorRole
            and col == 0
            and item.kind == ItemKind.COMPONENT
            and item.component_id is not None
        ):
            kind = item.component_kind or "peak"
            return color_for_component(kind=kind, index=item.color_index)

        if (
            role == Qt.ItemDataRole.CheckStateRole
            and col == 1
            and item.kind == ItemKind.PARAMETER_FIELD
            and item.parameter_field == "vary"
        ):
            return Qt.CheckState.Checked if bool(item.value) else Qt.CheckState.Unchecked

        if (
            role == Qt.ItemDataRole.ForegroundRole
            and col == 0
            and item.kind == ItemKind.PARAMETER_FIELD
        ):
            return QColor(theme.TEXT_SUBTLE)

        if role == Qt.ItemDataRole.ForegroundRole and col == 0 and item.kind == ItemKind.ACTION_ROW:
            return QColor(theme.TEXT_MUTED)

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if col == 0:
                # EditRole uses the stored name (or empty) so positional fallbacks
                # like "Peak 1" are not written back on a no-op edit.
                if role == Qt.ItemDataRole.EditRole and item.kind == ItemKind.COMPONENT:
                    return item.stored_name or ""
                if item.kind == ItemKind.PARAMETER_ROW and role == Qt.ItemDataRole.DisplayRole:
                    marks = _parameter_constraint_marks(item)
                    return f"{item.name}  {marks}" if marks else item.name
                return item.name
            if item.kind == ItemKind.ACTION_ROW:
                return None
            if item.kind == ItemKind.PARAMETER_ROW and col == 1:
                return _format_value(item.value)
            if item.kind == ItemKind.PARAMETER_FIELD and col == 1:
                if item.parameter_field == "vary":
                    return ""
                if item.parameter_field == "expr":
                    return "" if item.value is None else str(item.value)
                return _format_value(item.value)
            if item.kind == ItemKind.REGION_SLICE and col == 1:
                return _format_value(item.value)
            if item.kind == ItemKind.COMPONENT_MODEL and col == 1:
                return _format_value(item.value)
            if _is_area_item(item) and col == 1:
                return _area_label(item.value)

        return None

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        """Return item flags, including editable and checkable columns."""
        item = self._item(index)
        if not isinstance(item, PropertyItem):
            return Qt.ItemFlag.NoItemFlags

        if item.kind == ItemKind.ACTION_ROW:
            return Qt.ItemFlag.ItemIsEnabled

        base_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        col = index.column()

        if item.kind == ItemKind.PARAMETER_ROW:
            if col == 1:
                return base_flags | Qt.ItemFlag.ItemIsEditable
            return base_flags

        if item.kind == ItemKind.PARAMETER_FIELD and col == 1:
            if item.parameter_field == "vary":
                return base_flags | Qt.ItemFlag.ItemIsUserCheckable
            if item.parameter_field == "expr":
                # Edited via ExprEditorPopup on click, not an inline QLineEdit.
                return base_flags
            return base_flags | Qt.ItemFlag.ItemIsEditable

        if col == 0 and item.kind == ItemKind.COMPONENT and item.component_id is not None:
            return base_flags | Qt.ItemFlag.ItemIsEditable

        if col == 1 and item.kind == ItemKind.REGION_SLICE:
            return base_flags | Qt.ItemFlag.ItemIsEditable

        return base_flags

    def _add_region_slice(
        self,
        parent_item: PropertyItem,
        region_id: str,
        start_val: int | float,
        stop_val: int | float,
        *,
        slice_mode: Literal["value", "index"],
    ) -> None:
        """Add start and stop rows under parent for the given region slice."""
        soft_lo, soft_hi = self._slice_soft_range(region_id, slice_mode)
        parent_item.append_child(
            PropertyItem(
                name="start",
                value=start_val,
                parent=parent_item,
                kind=ItemKind.REGION_SLICE,
                region_id=region_id,
                soft_lo=soft_lo,
                soft_hi=soft_hi,
            )
        )
        parent_item.append_child(
            PropertyItem(
                name="stop",
                value=stop_val,
                parent=parent_item,
                kind=ItemKind.REGION_SLICE,
                region_id=region_id,
                soft_lo=soft_lo,
                soft_hi=soft_hi,
            )
        )

    def _slice_soft_range(
        self,
        region_id: str,
        slice_mode: Literal["value", "index"],
    ) -> tuple[float, float]:
        """Return soft slider bounds for region start/stop rows."""
        x_min: float | None = None
        x_max: float | None = None
        index_count: int | None = None
        try:
            spectrum_id = self._controller.query.get_parent_id(region_id)
            spectrum = self._controller.query.get_spectrum_dto(spectrum_id, normalized=False)
            if spectrum.x.size:
                x_min = float(np.min(spectrum.x))
                x_max = float(np.max(spectrum.x))
                index_count = int(spectrum.x.size)
        except KeyError:
            pass
        return soft_region_bound_range(
            mode=slice_mode,
            x_min=x_min,
            x_max=x_max,
            index_count=index_count,
        )

    def _add_parameters(
        self,
        parent_item: PropertyItem,
        component_id: str,
        parameters_dto: dict[str, Any],
        *,
        model_name: str,
        x_min: float | None = None,
        x_max: float | None = None,
        y_max: float | None = None,
    ) -> None:
        """Add one parameter row with nested lower / upper / vary / expr children."""
        model = ModelRegistry.get(model_name)
        for param_name, param_dto in parameters_dto.items():
            soft_lo, soft_hi = model.soft_parameter_range(
                str(param_name),
                float(param_dto.value),
                float(param_dto.lower),
                float(param_dto.upper),
                x_min=x_min,
                x_max=x_max,
                y_max=y_max,
            )
            param_item = PropertyItem(
                name=str(param_name),
                value=param_dto.value,
                parent=parent_item,
                kind=ItemKind.PARAMETER_ROW,
                component_id=component_id,
                parameter_name=str(param_name),
                soft_lo=soft_lo,
                soft_hi=soft_hi,
                param_lower=param_dto.lower,
                param_upper=param_dto.upper,
                param_vary=bool(param_dto.vary),
                param_expr=param_dto.expr,
            )
            parent_item.append_child(param_item)
            for field_name, field_value in (
                ("lower", param_dto.lower),
                ("upper", param_dto.upper),
                ("vary", bool(param_dto.vary)),
                ("expr", param_dto.expr),
            ):
                param_item.append_child(
                    PropertyItem(
                        name=field_name,
                        value=field_value,
                        parent=param_item,
                        kind=ItemKind.PARAMETER_FIELD,
                        component_id=component_id,
                        parameter_name=str(param_name),
                        parameter_field=field_name,  # type: ignore[arg-type]
                    )
                )

    def _region_soft_context(
        self,
        region_id: str,
    ) -> tuple[float | None, float | None, float | None]:
        """Return ``(x_min, x_max, y_max)`` hints for soft slider ranges."""
        try:
            region_dto = self._controller.query.get_region_dto(region_id, normalized=False)
        except KeyError:
            return None, None, None
        x = region_dto.x
        y = region_dto.y
        if x.size == 0:
            return None, None, None
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        y_max = float(np.max(y)) if y.size else None
        return x_min, x_max, y_max

    def setData(
        self,
        index: QModelIndex | QPersistentModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        """Write an edited value back to the controller and refresh the item."""
        item = self._item(index)
        if not isinstance(item, PropertyItem):
            return False

        col = index.column()

        if role == Qt.ItemDataRole.CheckStateRole:
            if (
                item.kind == ItemKind.PARAMETER_FIELD
                and col == 1
                and item.parameter_field == "vary"
                and item.component_id is not None
                and item.parameter_name is not None
            ):
                coerced = value == Qt.CheckState.Checked.value
                self._controller.update_parameter(
                    item.component_id,
                    item.parameter_name,
                    "vary",
                    coerced,
                    normalized=False,
                )
                item.value = coerced
                parent_param = item.parent
                if parent_param is not None and parent_param.kind == ItemKind.PARAMETER_ROW:
                    parent_param.param_vary = coerced
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
                return True
            return False

        if role != Qt.ItemDataRole.EditRole:
            return False

        if item.kind == ItemKind.COMPONENT and col == 0 and item.component_id is not None:
            new_name = str(value).strip() or None
            if new_name == item.stored_name:
                return True
            self._controller.rename_component(item.component_id, new_name)
            self.refresh()
            return True

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
            self.dataChanged.emit(
                index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole]
            )
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
            and col == 1
        ):
            text = str(value).strip().replace(",", ".")
            try:
                coerced: str | float | None = float(text)
            except ValueError:
                return False
            self._controller.update_parameter(
                item.component_id,
                item.parameter_name,
                "value",
                coerced,
                normalized=False,
            )
            item.value = coerced
            self._recompute_soft_range(item)
            self.dataChanged.emit(
                index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole]
            )
            return True

        if (
            item.kind == ItemKind.PARAMETER_FIELD
            and col == 1
            and item.component_id is not None
            and item.parameter_name is not None
            and item.parameter_field is not None
        ):
            field = item.parameter_field
            coerced_field: str | float | None
            if field == "expr":
                if value is None:
                    coerced_field = None
                else:
                    coerced_field = str(value).strip() or None
            elif field in {"lower", "upper"}:
                text = str(value).strip().replace(",", ".")
                if text in {"", "—", "-", "inf", "+inf", "-inf", "∞", "-∞"}:
                    coerced_field = float("-inf") if field == "lower" else float("inf")
                else:
                    try:
                        coerced_field = float(text)
                    except ValueError:
                        return False
            else:
                return False
            self._controller.update_parameter(
                item.component_id,
                item.parameter_name,
                field,
                coerced_field,
                normalized=False,
            )
            item.value = coerced_field
            parent_param = item.parent
            if parent_param is not None and parent_param.kind == ItemKind.PARAMETER_ROW:
                if field == "lower":
                    parent_param.param_lower = coerced_field
                elif field == "upper":
                    parent_param.param_upper = coerced_field
                else:
                    parent_param.param_expr = coerced_field
                if field in {"lower", "upper"}:
                    self._recompute_soft_range(parent_param)
                self._emit_parameter_row_changed(parent_param)
            else:
                self.dataChanged.emit(
                    index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole]
                )
            return True

        return False

    def _emit_parameter_row_changed(self, item: PropertyItem) -> None:
        """Emit dataChanged for a parameter row and its nested field children."""
        parent = item.parent
        if parent is None:
            return
        row = item.row()
        if row < 0:
            return
        parent_index = (
            QModelIndex()
            if parent is self._root_item
            else self.createIndex(parent.row(), 0, parent)
        )
        param_index = self.index(row, 0, parent_index)
        right = self.index(row, self.columnCount() - 1, parent_index)
        self.dataChanged.emit(
            param_index,
            right,
            [
                Qt.ItemDataRole.DisplayRole,
                Qt.ItemDataRole.EditRole,
                Qt.ItemDataRole.CheckStateRole,
            ],
        )
        for child_row, child in enumerate(item.children):
            if child.kind != ItemKind.PARAMETER_FIELD:
                continue
            if child.parameter_field == "lower":
                child.value = item.param_lower
            elif child.parameter_field == "upper":
                child.value = item.param_upper
            elif child.parameter_field == "expr":
                child.value = item.param_expr
            elif child.parameter_field == "vary":
                child.value = item.param_vary
            cleft = self.index(child_row, 0, param_index)
            cright = self.index(child_row, self.columnCount() - 1, param_index)
            roles = [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole]
            if child.parameter_field == "vary":
                roles.append(Qt.ItemDataRole.CheckStateRole)
            self.dataChanged.emit(cleft, cright, roles)

    def sync_parameter_item_from_controller(self, item: PropertyItem) -> None:
        """
        Refresh a parameter row's cached fields from the live component DTO.

        Used after slider commits so soft ranges and nested field rows stay current
        without a full tree rebuild.
        """
        if (
            item.kind != ItemKind.PARAMETER_ROW
            or item.component_id is None
            or item.parameter_name is None
        ):
            return
        try:
            dto = self._controller.query.get_component_dto(item.component_id)
            param = dto.parameters[item.parameter_name]
        except KeyError:
            return
        item.value = param.value
        item.param_lower = param.lower
        item.param_upper = param.upper
        item.param_vary = bool(param.vary)
        item.param_expr = param.expr
        self._recompute_soft_range(item)
        self._emit_parameter_row_changed(item)

    def _recompute_soft_range(self, item: PropertyItem) -> None:
        """Update ``soft_lo`` / ``soft_hi`` for a parameter row from model soft ranges."""
        if item.parameter_name is None or item.component_id is None:
            return
        x_min = x_max = y_max = None
        region_id = item.region_id
        if region_id is None:
            try:
                region_id = self._controller.query.get_parent_id(item.component_id)
            except KeyError:
                region_id = None
        if region_id is not None:
            x_min, x_max, y_max = self._region_soft_context(region_id)
        try:
            model_name = self._controller.query.get_component_dto(item.component_id).model.name
        except KeyError:
            return
        item.soft_lo, item.soft_hi = ModelRegistry.get(model_name).soft_parameter_range(
            item.parameter_name,
            float(item.value) if isinstance(item.value, (int, float)) else 0.0,
            float(item.param_lower)
            if isinstance(item.param_lower, (int, float))
            else float("-inf"),
            float(item.param_upper) if isinstance(item.param_upper, (int, float)) else float("inf"),
            x_min=x_min,
            x_max=x_max,
            y_max=y_max,
        )

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
        if spectrum_id is None or not self._controller.query.check_object_exists(spectrum_id):
            self._root_item.append_child(
                PropertyItem(name="No spectrum selected", parent=self._root_item)
            )
            self.endResetModel()
            return

        query = self._controller.query
        params = self._controller.get_app_parameters()
        slice_mode = params.region_slice_display_mode

        region_ids = query.get_regions_ids(spectrum_id)
        for idx, region_id in enumerate(region_ids, start=1):
            region_item = PropertyItem(
                name=f"Region {idx}",
                parent=self._root_item,
                kind=ItemKind.REGION,
                region_id=region_id,
                object_id=region_id,
            )
            self._root_item.append_child(region_item)

            start_val, stop_val = query.get_region_slice(region_id, mode=slice_mode)
            start_val = (
                start_val if start_val is not None else (0 if slice_mode == "index" else 0.0)
            )
            stop_val = stop_val if stop_val is not None else (0 if slice_mode == "index" else 0.0)
            self._add_region_slice(
                region_item, region_id, start_val, stop_val, slice_mode=slice_mode
            )

            background_id = query.get_background_id(region_id)
            peaks_ids = list(query.get_peaks_ids(region_id))
            x_min, x_max, y_max = self._region_soft_context(region_id)
            region_item.value = query.get_region_area(region_id)

            if background_id is not None:
                background_dto = query.get_component_dto(background_id)
                background_item = PropertyItem(
                    name=background_dto.name or "Background",
                    parent=region_item,
                    kind=ItemKind.COMPONENT,
                    region_id=region_id,
                    component_id=background_id,
                    component_kind="background",
                    object_id=background_id,
                    stored_name=background_dto.name,
                )
                region_item.append_child(background_item)
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
                self._add_parameters(
                    background_item,
                    background_id,
                    background_dto.parameters,
                    model_name=background_dto.model.name,
                    x_min=x_min,
                    x_max=x_max,
                    y_max=y_max,
                )
            else:
                region_item.append_child(
                    PropertyItem(
                        name="Add background",
                        parent=region_item,
                        kind=ItemKind.ACTION_ROW,
                        action="add_background",
                        region_id=region_id,
                    )
                )

            for peak_index, peak_id in enumerate(peaks_ids, start=1):
                peak_dto = query.get_component_dto(peak_id)
                peak_item = PropertyItem(
                    name=peak_dto.name or f"Peak {peak_index}",
                    value=query.get_peak_area(peak_id),
                    parent=region_item,
                    kind=ItemKind.COMPONENT,
                    region_id=region_id,
                    component_id=peak_id,
                    component_kind="peak",
                    object_id=peak_id,
                    stored_name=peak_dto.name,
                    color_index=peak_index - 1,
                )
                region_item.append_child(peak_item)
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
                self._add_parameters(
                    peak_item,
                    peak_id,
                    peak_dto.parameters,
                    model_name=peak_dto.model.name,
                    x_min=x_min,
                    x_max=x_max,
                    y_max=y_max,
                )

            region_item.append_child(
                PropertyItem(
                    name="Add peak",
                    parent=region_item,
                    kind=ItemKind.ACTION_ROW,
                    action="add_peak",
                    region_id=region_id,
                )
            )

        self._root_item.append_child(
            PropertyItem(
                name="Add region",
                parent=self._root_item,
                kind=ItemKind.ACTION_ROW,
                action="add_region",
            )
        )

        self.endResetModel()


class PropertiesDelegate(QStyledItemDelegate):
    """Delegate for model chips, soft-range editors, and hover copy icons."""

    _COPY_FEEDBACK_MS = 1200

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._hovered_index = QPersistentModelIndex()
        self._hovered_copy = False
        self._copied_index = QPersistentModelIndex()
        self._copy_feedback_token = 0

    @staticmethod
    def _shows_copy(index: QModelIndex | QPersistentModelIndex) -> bool:
        item = index.internalPointer() if index.isValid() else None
        return (
            isinstance(item, PropertyItem) and index.column() == 1 and _is_copyable_value_item(item)
        )

    def _has_copy_feedback(self, index: QModelIndex | QPersistentModelIndex) -> bool:
        return self._copied_index.isValid() and _as_model_index(
            self._copied_index
        ) == _as_model_index(index)

    def flash_copy_success(self, index: QModelIndex) -> None:
        """Briefly show a check icon on ``index`` after a successful copy."""
        if not index.isValid() or not self._shows_copy(index):
            return
        view = self.parent()
        old = _as_model_index(self._copied_index)
        self._copied_index = QPersistentModelIndex(index)
        self._copy_feedback_token += 1
        token = self._copy_feedback_token
        if isinstance(view, QTreeView):
            if old.isValid() and old != index:
                view.update(old)
            view.update(index)

        def _clear() -> None:
            if token != self._copy_feedback_token:
                return
            cleared = _as_model_index(self._copied_index)
            self._copied_index = QPersistentModelIndex()
            parent = self.parent()
            if isinstance(parent, QTreeView) and cleared.isValid():
                parent.update(cleared)

        QTimer.singleShot(self._COPY_FEEDBACK_MS, _clear)

    def set_hover(
        self,
        index: QModelIndex | QPersistentModelIndex,
        pos: QPoint | None = None,
    ) -> None:
        """Update hover target and whether the copy glyph is under the cursor."""
        view = self.parent()
        old = _as_model_index(self._hovered_index)
        new = (
            _as_model_index(index) if index.isValid() and self._shows_copy(index) else QModelIndex()
        )
        hovered_copy = False
        if new.isValid() and pos is not None and isinstance(view, QTreeView):
            option = QStyleOptionViewItem()
            option.rect = view.visualRect(new)
            hovered_copy = _action_icon_rect(option).contains(pos)
        changed = old != new or hovered_copy != self._hovered_copy
        self._hovered_index = (
            QPersistentModelIndex(new) if new.isValid() else QPersistentModelIndex()
        )
        self._hovered_copy = hovered_copy
        if changed and isinstance(view, QTreeView):
            if old.isValid():
                view.update(old)
            if new.isValid():
                view.update(new)

    def clear_hover(self) -> None:
        """Clear hover highlighting."""
        self.set_hover(QModelIndex(), None)

    def hit_copy(
        self,
        index: QModelIndex | QPersistentModelIndex,
        pos: QPoint,
        visual_rect: QRect,
    ) -> bool:
        """Return True when ``pos`` is over the copy glyph of a copyable cell."""
        if not self._shows_copy(index):
            return False
        option = QStyleOptionViewItem()
        option.rect = visual_rect
        return _action_icon_rect(option).contains(pos)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """
        Paint cells; draw model chips, suppress text under open editors, copy icon.

        Model values always look like compact Cursor-style controls. Persistent
        slider editors otherwise sit on top of DisplayRole text. Copy appears on
        hover for value / lower / upper / expr cells; briefly becomes a check
        after a successful copy.
        """
        item = index.internalPointer() if index.isValid() else None
        view = self.parent()
        has_editor = (
            isinstance(view, QTreeView)
            and index.isValid()
            and view.indexWidget(_as_model_index(index)) is not None
        )
        hovered_row = self._hovered_index.isValid() and _as_model_index(
            self._hovered_index
        ) == _as_model_index(index)
        feedback = self._has_copy_feedback(index)
        show_copy = (
            isinstance(item, PropertyItem) and self._shows_copy(index) and (hovered_row or feedback)
        )

        if (
            has_editor
            and isinstance(item, PropertyItem)
            and item.kind
            in {ItemKind.PARAMETER_ROW, ItemKind.REGION_SLICE, ItemKind.PARAMETER_FIELD}
            and index.column() == 1
        ):
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            opt.text = ""
            widget = opt.widget
            style = widget.style() if widget is not None else None
            if style is not None:
                style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
            if show_copy:
                self._paint_copy_icon(painter, option, index)
            return

        if (
            isinstance(item, PropertyItem)
            and item.kind == ItemKind.COMPONENT_MODEL
            and index.column() == 1
        ):
            self._paint_model_chip(painter, option, index, str(item.value or ""))
            return

        if isinstance(item, PropertyItem) and _is_area_item(item) and index.column() == 1:
            self._paint_area_label(painter, option, index, show_copy=show_copy)
            return

        if show_copy:
            reserve = _action_reserve_width()
            text_option = QStyleOptionViewItem(option)
            text_option.rect = option.rect.adjusted(0, 0, -reserve, 0)
            super().paint(painter, text_option, index)
            strip = QRect(
                option.rect.right() - reserve + 1,
                option.rect.y(),
                reserve,
                option.rect.height(),
            )
            widget = option.widget
            style = widget.style() if widget is not None else None
            if style is not None:
                strip_option = QStyleOptionViewItem(option)
                self.initStyleOption(strip_option, index)
                strip_option.text = ""
                painter.save()
                painter.setClipRect(strip)
                style.drawControl(
                    QStyle.ControlElement.CE_ItemViewItem,
                    strip_option,
                    painter,
                    widget,
                )
                painter.restore()
            self._paint_copy_icon(painter, option, index)
            return

        super().paint(painter, option, index)

    def _paint_area_label(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
        *,
        show_copy: bool,
    ) -> None:
        """Draw ``s = …`` in italic, using the same gray as truncated IDs."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        label = opt.text
        opt.text = ""
        widget = opt.widget
        style = widget.style() if widget is not None else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)

        if label:
            painter.save()
            font = opt.font
            font.setItalic(True)
            painter.setFont(font)
            if opt.state & QStyle.StateFlag.State_Selected:
                selected = QColor(opt.palette.highlightedText().color())
                selected.setAlpha(180)
                painter.setPen(selected)
            else:
                painter.setPen(QColor(ID_SUFFIX_HEX))
            text_rect = opt.rect
            if style is not None:
                candidate = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, widget)
                if candidate.isValid():
                    text_rect = candidate
            if show_copy:
                text_rect = text_rect.adjusted(0, 0, -_action_reserve_width(), 0)
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            painter.restore()

        if show_copy:
            self._paint_copy_icon(painter, option, index)

    def _paint_copy_icon(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Draw copy or brief check feedback on the trailing action slot."""
        copy_rect = _action_icon_rect(option)
        if self._has_copy_feedback(index):
            _paint_action_icon(painter, _CHECK_ICON, copy_rect, active=True)
        else:
            _paint_action_icon(painter, _COPY_ICON, copy_rect, active=self._hovered_copy)

    def updateEditorGeometry(
        self,
        editor: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Leave room on the right for the hover copy icon on copyable cells."""
        if self._shows_copy(index):
            editor.setGeometry(option.rect.adjusted(0, 0, -_action_reserve_width(), 0))
            return
        super().updateEditorGeometry(editor, option, index)

    def _paint_model_chip(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
        text: str,
    ) -> None:
        """Draw a compact Cursor-style value chip that does not change row height."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        widget = opt.widget
        style = widget.style() if widget is not None else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        chip_h = min(22, max(16, opt.rect.height() - 6))
        y = opt.rect.y() + (opt.rect.height() - chip_h) // 2
        rect = QRect(opt.rect.x() + 4, y, max(0, opt.rect.width() - 12), chip_h)
        if rect.width() < 24:
            painter.restore()
            return

        painter.setPen(QColor(theme.BORDER))
        painter.setBrush(QColor(theme.SURFACE))
        painter.drawRoundedRect(rect, 6, 6)

        chevron = "▾"
        metrics = opt.fontMetrics
        chevron_w = metrics.horizontalAdvance(chevron) + 8
        text_rect = rect.adjusted(8, 0, -chevron_w, 0)
        painter.setPen(QColor(theme.TEXT))
        painter.setFont(opt.font)
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            elided,
        )
        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.drawText(
            rect.adjusted(0, 0, -6, 0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            chevron,
        )
        painter.restore()

    def show_model_menu(self, global_pos: QPoint, index: QModelIndex) -> None:
        """
        Show a Cursor-style model picker menu for a ``COMPONENT_MODEL`` cell.

        Parameters
        ----------
        global_pos : QPoint
            Global position for the popup.
        index : QModelIndex
            Model value cell index.
        """
        item = index.internalPointer() if index.isValid() else None
        if not isinstance(item, PropertyItem) or item.kind != ItemKind.COMPONENT_MODEL:
            return
        view = self.parent()
        if not isinstance(view, QWidget):
            return

        if item.component_kind == "peak":
            names = self._controller.query.get_peak_model_names()
        else:
            names = self._controller.query.get_background_model_names()
        current = str(item.value or "")

        menu = QMenu(view)
        for name in names:
            action = menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.setData(name)

        menu.move(global_pos)
        exec_menu: Any = menu.exec
        chosen = exec_menu()
        if chosen is None:
            return
        new_name = str(chosen.data())
        if not new_name or new_name == current:
            return
        model = index.model()
        if model is not None:
            model.setData(index, new_name, Qt.ItemDataRole.EditRole)

    def show_expr_popup(self, index: QModelIndex) -> None:
        """
        Open the expression constructor popup for a ``PARAMETER_FIELD`` expr cell.

        Parameters
        ----------
        index : QModelIndex
            Expr value cell index.
        """
        item = index.internalPointer() if index.isValid() else None
        if (
            not isinstance(item, PropertyItem)
            or item.kind != ItemKind.PARAMETER_FIELD
            or item.parameter_field != "expr"
        ):
            return
        view = self.parent()
        if not isinstance(view, QWidget):
            return

        region_id = _ancestor_region_id(item)
        spectrum_id = getattr(self._controller, "selected_spectrum_id", None)
        if isinstance(spectrum_id, str) and not spectrum_id:
            spectrum_id = None
        if spectrum_id is None and region_id is not None:
            try:
                spectrum_id = self._controller.query.get_parent_id(region_id)
            except KeyError:
                spectrum_id = None

        initial = "" if item.value is None else str(item.value)
        persistent = QPersistentModelIndex(index)

        def _on_accepted(text: object) -> None:
            model = persistent.model()
            if model is None or not persistent.isValid():
                return
            model.setData(_as_model_index(persistent), text, Qt.ItemDataRole.EditRole)

        ExprEditorPopup.open_for(
            self._controller,
            initial_text=initial,
            focus_spectrum_id=spectrum_id if isinstance(spectrum_id, str) else None,
            focus_region_id=region_id,
            parent=view,
            on_accepted=_on_accepted,
        )

    def createEditor(
        self,
        parent: QWidget,
        option: Any,
        index: QModelIndex | QPersistentModelIndex,
    ) -> QWidget:
        """Create a soft-range slider editor for parameter/slice value cells."""
        if index.column() != 1:
            return super().createEditor(parent, option, index)
        item = index.internalPointer() if index.isValid() else None
        if not isinstance(item, PropertyItem):
            return super().createEditor(parent, option, index)

        if (
            item.kind == ItemKind.PARAMETER_ROW
            and item.component_id is not None
            and item.parameter_name is not None
            and item.soft_lo is not None
            and item.soft_hi is not None
        ):
            editor = ParameterValueEditor(
                self._controller,
                component_id=item.component_id,
                parameter_name=item.parameter_name,
                soft_lo=item.soft_lo,
                soft_hi=item.soft_hi,
                parent=parent,
            )

            def _commit_param() -> None:
                self.commitData.emit(editor)

            editor.editingFinished.connect(_commit_param)
            return editor

        if (
            item.kind == ItemKind.REGION_SLICE
            and item.region_id is not None
            and item.name in {"start", "stop"}
            and item.soft_lo is not None
            and item.soft_hi is not None
        ):
            slice_mode = self._controller.get_app_parameters().region_slice_display_mode
            editor = ParameterValueEditor(
                self._controller,
                region_id=item.region_id,
                slice_bound=item.name,  # type: ignore[arg-type]
                slice_mode=slice_mode,
                soft_lo=item.soft_lo,
                soft_hi=item.soft_hi,
                parent=parent,
            )

            def _commit_slice() -> None:
                self.commitData.emit(editor)

            editor.editingFinished.connect(_commit_slice)
            return editor

        if item.kind == ItemKind.PARAMETER_FIELD and item.parameter_field in {
            "lower",
            "upper",
        }:
            editor = QLineEdit(parent)
            editor.setFrame(False)
            editor.setObjectName("PropertiesFieldEdit")
            return editor

        return super().createEditor(parent, option, index)

    def setEditorData(self, editor: QWidget, index: QModelIndex | QPersistentModelIndex) -> None:
        """Populate the editor with the current parameter or slice value."""
        if index.column() != 1:
            super().setEditorData(editor, index)
            return
        item = index.internalPointer() if index.isValid() else None
        if not isinstance(item, PropertyItem):
            super().setEditorData(editor, index)
            return

        if item.kind in {ItemKind.PARAMETER_ROW, ItemKind.REGION_SLICE} and isinstance(
            editor, ParameterValueEditor
        ):
            editor.set_value(float(item.value))
            return

        if (
            item.kind == ItemKind.PARAMETER_FIELD
            and isinstance(editor, QLineEdit)
            and item.parameter_field in {"lower", "upper"}
        ):
            editor.setText(_format_value(item.value) if item.value is not None else "")
            # Show empty for ±inf so the cell matches collapsed display.
            if isinstance(item.value, (int, float)) and not math.isfinite(float(item.value)):
                editor.setText("")
            return

        super().setEditorData(editor, index)

    def setModelData(
        self,
        editor: QWidget,
        model: QAbstractItemModel,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Commit a parameter or region-slice value from the slider editor."""
        if index.column() != 1:
            super().setModelData(editor, model, index)
            return
        item = index.internalPointer() if index.isValid() else None
        if not isinstance(item, PropertyItem):
            super().setModelData(editor, model, index)
            return

        if item.kind in {ItemKind.PARAMETER_ROW, ItemKind.REGION_SLICE} and isinstance(
            editor, ParameterValueEditor
        ):
            editor.commit_if_needed()
            item.value = editor.value()
            top_left = model.index(index.row(), index.column(), index.parent())
            model.dataChanged.emit(
                top_left,
                top_left,
                [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole],
            )
            return

        super().setModelData(editor, model, index)

    def sizeHint(
        self,
        option: Any,
        index: QModelIndex | QPersistentModelIndex,
    ) -> Any:
        """Use a taller row only while the soft-range slider editor is open."""
        hint = super().sizeHint(option, index)
        view = self.parent()
        if not (isinstance(view, QTreeView) and index.isValid() and index.column() == 1):
            return hint
        widget = view.indexWidget(_as_model_index(index))
        if isinstance(widget, ParameterValueEditor):
            hint.setHeight(max(hint.height(), 52))
            hint.setWidth(max(hint.width(), 100))
        return hint

    def destroyEditor(
        self,
        editor: QWidget,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Restore previewed values when the editor is cancelled."""
        if isinstance(editor, ParameterValueEditor) and not editor.committed:
            editor.cancel_preview()
        super().destroyEditor(editor, index)


_ACTION_BTN_SIZE = 14
_ACTION_BTN_MARGIN = 4
_ACTION_BTN_GAP = 4
_ACTION_ICON_GAP = 6
_PLUS_ICON = QIcon(str(icon_path("plus.svg")))
_X_ICON = QIcon(str(icon_path("x.svg")))
_COPY_ICON = QIcon(str(icon_path("copy.svg")))
_CHECK_ICON = QIcon(str(icon_path("check.svg")))
_OPTIMIZE_ICON = QIcon(str(icon_path("optimize.svg")))


def _area_label(value: Any) -> str:
    """Return the read-only area caption, or an empty string when unset."""
    formatted = _format_value(value)
    if not formatted:
        return ""
    return f"s = {formatted}"


def _is_area_item(item: PropertyItem) -> bool:
    """Return True for region and peak rows that show a read-only fitted area."""
    if item.kind == ItemKind.REGION:
        return True
    return item.kind == ItemKind.COMPONENT and item.component_kind == "peak"


def _is_copyable_value_item(item: PropertyItem) -> bool:
    """Return True for value / lower / upper / expr / area cells that support copy."""
    if item.kind == ItemKind.PARAMETER_ROW:
        return True
    if _is_area_item(item) and item.value is not None:
        return True
    return item.kind == ItemKind.PARAMETER_FIELD and item.parameter_field in {
        "lower",
        "upper",
        "expr",
    }


def _clipboard_text_for_item(item: PropertyItem) -> str:
    """Return the clipboard string for a copyable parameter value cell."""
    if item.kind == ItemKind.PARAMETER_FIELD and item.parameter_field == "expr":
        return "" if item.value is None else str(item.value)
    if item.value is None:
        return ""
    if isinstance(item.value, bool):
        return str(item.value)
    if isinstance(item.value, (int, float)):
        return _format_value(item.value)
    return str(item.value)


def _action_icon_rect_at(option: QStyleOptionViewItem, slot_from_right: int) -> QRect:
    """
    Return an action-icon rect for the given right-aligned slot.

    Parameters
    ----------
    option : QStyleOptionViewItem
        Cell style option providing the paint rect.
    slot_from_right : int
        0 = rightmost icon, 1 = one slot left of that, etc.
    """
    size = _ACTION_BTN_SIZE
    y = option.rect.y() + max(0, (option.rect.height() - size) // 2)
    x = (
        option.rect.right()
        - _ACTION_BTN_MARGIN
        - size
        + 1
        - slot_from_right * (size + _ACTION_BTN_GAP)
    )
    return QRect(x, y, size, size)


def _action_icon_rect(option: QStyleOptionViewItem) -> QRect:
    """Return a trailing action-icon rect aligned to the right of the cell."""
    return _action_icon_rect_at(option, 0)


def _action_reserve_width(n_icons: int = 1) -> int:
    """Return horizontal space reserved for ``n_icons`` trailing action icons."""
    if n_icons <= 0:
        return 0
    return _ACTION_BTN_MARGIN + n_icons * _ACTION_BTN_SIZE + (n_icons - 1) * _ACTION_BTN_GAP + 2


def _paint_action_icon(
    painter: QPainter,
    icon: QIcon,
    rect: QRect,
    *,
    active: bool = False,
) -> None:
    """Paint an action icon without selection chrome (avoids framed Selected mode)."""
    painter.save()
    if active:
        painter.setOpacity(1.0)
    else:
        painter.setOpacity(0.72)
    icon.paint(painter, rect, Qt.AlignmentFlag.AlignCenter, QIcon.Mode.Normal)
    painter.restore()


class PropertiesNameDelegate(NameWithIdDelegate):
    """
    Name delegate for action rows (+ Add ...) and hover row actions.

    ``ACTION_ROW`` paints a plus icon and label. ``REGION`` rows show optimize
    + delete on hover; ``COMPONENT`` rows show delete only.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the delegate with empty hover state."""
        super().__init__(parent)
        self._hovered_index = QPersistentModelIndex()
        self._hovered_action: Literal["optimize", "delete"] | None = None

    @staticmethod
    def _shows_delete(index: QModelIndex | QPersistentModelIndex) -> bool:
        item = index.internalPointer() if index.isValid() else None
        return isinstance(item, PropertyItem) and item.kind in {
            ItemKind.REGION,
            ItemKind.COMPONENT,
        }

    @staticmethod
    def _shows_optimize(index: QModelIndex | QPersistentModelIndex) -> bool:
        item = index.internalPointer() if index.isValid() else None
        return isinstance(item, PropertyItem) and item.kind == ItemKind.REGION

    @staticmethod
    def _is_action_row(index: QModelIndex | QPersistentModelIndex) -> bool:
        item = index.internalPointer() if index.isValid() else None
        return isinstance(item, PropertyItem) and item.kind == ItemKind.ACTION_ROW

    @staticmethod
    def _action_icon_count(index: QModelIndex | QPersistentModelIndex) -> int:
        item = index.internalPointer() if index.isValid() else None
        if isinstance(item, PropertyItem) and item.kind == ItemKind.REGION:
            return 2
        if isinstance(item, PropertyItem) and item.kind == ItemKind.COMPONENT:
            return 1
        return 0

    @classmethod
    def _delete_rect(cls, option: QStyleOptionViewItem) -> QRect:
        """Return the delete-icon rect (rightmost action)."""
        return _action_icon_rect_at(option, 0)

    @classmethod
    def _optimize_rect(cls, option: QStyleOptionViewItem) -> QRect:
        """Return the optimize-icon rect (left of delete on region rows)."""
        return _action_icon_rect_at(option, 1)

    def set_hover(
        self,
        index: QModelIndex | QPersistentModelIndex,
        pos: QPoint | None = None,
    ) -> None:
        """Update hover target and which action glyph is under the cursor."""
        view = self.parent()
        shows_actions = self._shows_delete(index) or self._shows_optimize(index)
        old = _as_model_index(self._hovered_index)
        new = _as_model_index(index) if index.isValid() and shows_actions else QModelIndex()
        action: Literal["optimize", "delete"] | None = None
        if new.isValid() and pos is not None and isinstance(view, QTreeView):
            option = QStyleOptionViewItem()
            option.rect = view.visualRect(new)
            if self._shows_optimize(new) and self._optimize_rect(option).contains(pos):
                action = "optimize"
            elif self._shows_delete(new) and self._delete_rect(option).contains(pos):
                action = "delete"
        changed = old != new or action != self._hovered_action
        self._hovered_index = (
            QPersistentModelIndex(new) if new.isValid() else QPersistentModelIndex()
        )
        self._hovered_action = action
        if changed and isinstance(view, QTreeView):
            if old.isValid():
                view.update(old)
            if new.isValid():
                view.update(new)

    def clear_hover(self) -> None:
        """Clear hover highlighting."""
        self.set_hover(QModelIndex(), None)

    def hit_delete(
        self,
        index: QModelIndex | QPersistentModelIndex,
        pos: QPoint,
        visual_rect: QRect,
    ) -> bool:
        """Return True when ``pos`` is over the delete glyph of a deletable row."""
        if not self._shows_delete(index):
            return False
        option = QStyleOptionViewItem()
        option.rect = visual_rect
        return self._delete_rect(option).contains(pos)

    def hit_optimize(
        self,
        index: QModelIndex | QPersistentModelIndex,
        pos: QPoint,
        visual_rect: QRect,
    ) -> bool:
        """Return True when ``pos`` is over the optimize glyph of a region row."""
        if not self._shows_optimize(index):
            return False
        option = QStyleOptionViewItem()
        option.rect = visual_rect
        return self._optimize_rect(option).contains(pos)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Paint action-row plus+label, or name with hover row-action icons."""
        if self._is_action_row(index):
            self._paint_action_row(painter, option, index)
            return

        hovered_row = self._hovered_index.isValid() and _as_model_index(
            self._hovered_index
        ) == _as_model_index(index)
        n_icons = self._action_icon_count(index)
        if not (n_icons > 0 and hovered_row):
            super().paint(painter, option, index)
            return

        reserve = _action_reserve_width(n_icons)
        text_option = QStyleOptionViewItem(option)
        text_option.rect = option.rect.adjusted(0, 0, -reserve, 0)
        super().paint(painter, text_option, index)

        # Extend the same row chrome into the icon strip so selection does not
        # leave a lighter "frame" where text was clipped.
        strip = QRect(
            option.rect.right() - reserve + 1,
            option.rect.y(),
            reserve,
            option.rect.height(),
        )
        widget = option.widget
        style = widget.style() if widget is not None else None
        if style is not None:
            strip_option = QStyleOptionViewItem(option)
            self.initStyleOption(strip_option, index)
            strip_option.text = ""
            strip_option.icon = QIcon()
            strip_option.features &= ~QStyleOptionViewItem.ViewItemFeature.HasDecoration
            painter.save()
            painter.setClipRect(strip)
            style.drawControl(
                QStyle.ControlElement.CE_ItemViewItem,
                strip_option,
                painter,
                widget,
            )
            painter.restore()

        if self._shows_optimize(index):
            _paint_action_icon(
                painter,
                _OPTIMIZE_ICON,
                self._optimize_rect(option),
                active=self._hovered_action == "optimize",
            )
        if self._shows_delete(index):
            _paint_action_icon(
                painter,
                _X_ICON,
                self._delete_rect(option),
                active=self._hovered_action == "delete",
            )

    def _paint_action_row(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Draw selection chrome, plus icon, and muted action label."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        widget = opt.widget
        style = widget.style() if widget is not None else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)

        icon_size = _ACTION_BTN_SIZE
        y = option.rect.y() + max(0, (option.rect.height() - icon_size) // 2)
        icon_rect = QRect(option.rect.x() + 2, y, icon_size, icon_size)
        _paint_action_icon(painter, _PLUS_ICON, icon_rect, active=False)

        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        painter.save()
        painter.setPen(QColor(theme.TEXT_MUTED))
        text_rect = option.rect.adjusted(icon_size + _ACTION_ICON_GAP + 2, 0, 0, 0)
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            text,
        )
        painter.restore()

    def editorEvent(
        self,
        event: QEvent,
        model: QAbstractItemModel,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> bool:
        """Block default press handling on row-action glyphs (view owns them)."""
        if (
            event.type() == QEvent.Type.MouseButtonPress
            and isinstance(event, QMouseEvent)
            and event.button() == Qt.MouseButton.LeftButton
            and (
                (
                    self._shows_optimize(index)
                    and self._optimize_rect(option).contains(event.position().toPoint())
                )
                or (
                    self._shows_delete(index)
                    and self._delete_rect(option).contains(event.position().toPoint())
                )
            )
        ):
            return True
        return super().editorEvent(event, model, option, index)


class PropertiesView(EditorTreeView):
    """
    View used for the read-only Properties panel.

    The view owns a :class:`PropertiesModel` instance and exposes a convenience
    :meth:`refresh` method that can be connected directly to controller
    signals.
    """

    # Flexible columns 0,1 (Name, Value): Name is 1.5x Value.
    _FLEX_WEIGHT_NAME = 3
    _FLEX_WEIGHT_VALUE = 2

    def __init__(self, controller: ControllerWrapper, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._syncing_selection = False
        self._updating_from_view = False
        self._refresh_scheduled = False
        self._last_spectrum_id: str | None = None
        self._applied_default_expand = False
        self._slider_editor_index: QPersistentModelIndex | None = None
        self._expanded_param_index: QPersistentModelIndex | None = None
        self._suppress_cell_picker = False
        self._model = PropertiesModel(controller, self)
        self.setModel(self._model)
        self._name_delegate = PropertiesNameDelegate(self)
        self.setItemDelegateForColumn(0, self._name_delegate)
        self._value_delegate = PropertiesDelegate(controller, self)
        self.setItemDelegateForColumn(1, self._value_delegate)
        self.setIndentation(12)
        self.setHeaderHidden(False)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        hdr = self.header()
        hdr.setStretchLastSection(False)
        self._apply_column_widths()
        self.setUniformRowHeights(False)
        self.setAlternatingRowColors(False)
        apply_editor_tree_style(self)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_custom_context_menu)
        self.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self._refresh_now()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Run row actions/deletes/copy on press without changing the selection."""
        self._suppress_cell_picker = False
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            index = self.indexAt(pos)
            if index.isValid():
                name_index = (
                    index
                    if index.column() == 0
                    else self._model.index(index.row(), 0, index.parent())
                )
                value_index = (
                    index
                    if index.column() == 1
                    else self._model.index(index.row(), 1, index.parent())
                )
                item = name_index.internalPointer()
                if isinstance(item, PropertyItem) and item.kind == ItemKind.ACTION_ROW:
                    self._on_action_row_requested(name_index)
                    event.accept()
                    return
                if self._name_delegate.hit_optimize(name_index, pos, self.visualRect(name_index)):
                    self._on_row_optimize_requested(name_index)
                    event.accept()
                    return
                if self._name_delegate.hit_delete(name_index, pos, self.visualRect(name_index)):
                    self._on_row_delete_requested(name_index)
                    event.accept()
                    return
                if self._value_delegate.hit_copy(value_index, pos, self.visualRect(value_index)):
                    self._copy_value_at(value_index)
                    self._suppress_cell_picker = True
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Track hover for delete (name) and copy (value) glyphs."""
        pos = event.position().toPoint()
        index = self.indexAt(pos)
        name_index = QModelIndex()
        value_index = QModelIndex()
        if index.isValid():
            name_index = (
                index if index.column() == 0 else self._model.index(index.row(), 0, index.parent())
            )
            value_index = (
                index if index.column() == 1 else self._model.index(index.row(), 1, index.parent())
            )
        self._name_delegate.set_hover(name_index, pos)
        self._value_delegate.set_hover(value_index, pos)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        """Clear action-glyph hover when the pointer leaves the view."""
        self._name_delegate.clear_hover()
        self._value_delegate.clear_hover()
        super().leaveEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Keep flexible columns sized to the viewport with Name:Value = 1.5:1."""
        super().resizeEvent(event)
        self._update_flexible_column_widths()

    def drawBranches(
        self,
        painter: QPainter,
        rect: QRect,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Omit expand/collapse chevrons on parameter rows (auto-expanded on select)."""
        item = index.internalPointer() if index.isValid() else None
        if isinstance(item, PropertyItem) and item.kind in {
            ItemKind.PARAMETER_ROW,
            ItemKind.PARAMETER_FIELD,
        }:
            return
        super().drawBranches(painter, rect, index)

    def _apply_column_widths(self) -> None:
        """Lay out Name and Value across the full view width (Name 1.5x Value)."""
        hdr = self.header()
        for col in (0, 1):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
        self._update_flexible_column_widths()

    def _update_flexible_column_widths(self) -> None:
        """Divide viewport width across Name and Value using configured weights."""
        vp_w = int(self.viewport().width())
        if vp_w <= 0:
            return
        w_sum = self._FLEX_WEIGHT_NAME + self._FLEX_WEIGHT_VALUE
        w_name = (self._FLEX_WEIGHT_NAME * vp_w) // w_sum
        w_value = vp_w - w_name
        hdr = self.header()
        hdr.resizeSection(0, w_name)
        hdr.resizeSection(1, w_value)

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
        Schedule a tree rebuild after the current Qt event finishes.

        Deferring avoids resetting the model while a cell editor (e.g. lower /
        upper) is still leaving the edit stack, which can segfault in Qt.
        """
        if self._refresh_scheduled:
            return
        self._refresh_scheduled = True
        QTimer.singleShot(0, self._refresh_now)

    def _refresh_now(self) -> None:
        """Refresh tree contents; restore focused parameter/field when possible."""
        self._refresh_scheduled = False
        focus_key = self._stable_key_for_current_item()
        focus_column = 0
        current = self.selectionModel().currentIndex()
        if current.isValid():
            focus_column = current.column()
        self._close_parameter_slider_editor()
        self._expanded_param_index = None
        self._model.refresh()
        self._expand_hierarchy_except_param_details()
        self._applied_default_expand = True
        self._last_spectrum_id = self._controller.selected_spectrum_id
        if focus_key is not None and self._select_by_stable_key(focus_key, focus_column):
            pass
        else:
            self.sync_selection_from_controller()
        self._sync_parameter_detail_and_slider()

    def _expand_hierarchy_except_param_details(self) -> None:
        """Expand region/component nodes; leave parameter constraint children collapsed."""
        self.expandAll()
        root = QModelIndex()
        for r in range(self._model.rowCount(root)):
            region_index = self._model.index(r, 0, root)
            self._collapse_parameter_details_under(region_index)

    def _collapse_parameter_details_under(self, parent: QModelIndex) -> None:
        """Recursively collapse ``PARAMETER_ROW`` nodes (hide lower/upper/expr)."""
        for row in range(self._model.rowCount(parent)):
            child = self._model.index(row, 0, parent)
            item = child.internalPointer()
            if not isinstance(item, PropertyItem):
                continue
            if item.kind == ItemKind.PARAMETER_ROW:
                self.collapse(child)
            else:
                self._collapse_parameter_details_under(child)

    def _close_parameter_slider_editor(self) -> None:
        """Close the open parameter slider editor, if any."""
        if self._slider_editor_index is None:
            return
        idx = _as_model_index(self._slider_editor_index)
        if idx.isValid():
            widget = self.indexWidget(idx)
            if isinstance(widget, ParameterValueEditor):
                widget.commit_if_needed()
            self.closePersistentEditor(idx)
        self._slider_editor_index = None

    def _discard_parameter_slider_editor(self) -> None:
        """Close the slider editor without committing or restoring a preview."""
        if self._slider_editor_index is None:
            return
        idx = _as_model_index(self._slider_editor_index)
        if idx.isValid():
            widget = self.indexWidget(idx)
            if isinstance(widget, ParameterValueEditor):
                widget.abandon()
            self.closePersistentEditor(idx)
        self._slider_editor_index = None

    def _sync_parameter_detail_and_slider(self) -> None:
        """Expand selected parameter's constraints and open the soft-range slider."""
        current = self.selectionModel().currentIndex()
        target_index = QModelIndex()
        cursor = current
        while cursor.isValid():
            ptr = cursor.internalPointer()
            if isinstance(ptr, PropertyItem) and ptr.kind in {
                ItemKind.PARAMETER_ROW,
                ItemKind.REGION_SLICE,
            }:
                target_index = cursor
                break
            cursor = cursor.parent()

        prev_param = (
            _as_model_index(self._expanded_param_index)
            if self._expanded_param_index is not None
            else QModelIndex()
        )

        target_item = target_index.internalPointer() if target_index.isValid() else None
        is_param = (
            isinstance(target_item, PropertyItem) and target_item.kind == ItemKind.PARAMETER_ROW
        )

        if prev_param.isValid() and (not is_param or prev_param != target_index):
            self.collapse(prev_param)
            self._expanded_param_index = None

        if not target_index.isValid():
            self._close_parameter_slider_editor()
            self.doItemsLayout()
            return

        if is_param:
            self.expand(target_index)
            self._expanded_param_index = QPersistentModelIndex(target_index)

        value_index = self._model.index(target_index.row(), 1, target_index.parent())
        if (
            self._slider_editor_index is not None
            and _as_model_index(self._slider_editor_index) == value_index
        ):
            return

        self._close_parameter_slider_editor()
        self.openPersistentEditor(value_index)
        self._slider_editor_index = QPersistentModelIndex(value_index)
        self.doItemsLayout()

    def _sync_parameter_slider_editor(self) -> None:
        """Compatibility wrapper: sync detail expand + slider together."""
        self._sync_parameter_detail_and_slider()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Open model or expression pickers when their value cells are clicked."""
        super().mouseReleaseEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._suppress_cell_picker:
            self._suppress_cell_picker = False
            return
        index = self.indexAt(event.position().toPoint())
        if not index.isValid() or index.column() != 1:
            return
        item = index.internalPointer()
        if not isinstance(item, PropertyItem):
            return
        # Copy glyph shares the value cell; do not open pickers when releasing on it.
        if self._value_delegate.hit_copy(index, event.position().toPoint(), self.visualRect(index)):
            return
        rect = self.visualRect(index)
        global_pos = self.viewport().mapToGlobal(rect.bottomLeft())
        if item.kind == ItemKind.COMPONENT_MODEL:
            self._value_delegate.show_model_menu(global_pos, index)
            return
        if item.kind == ItemKind.PARAMETER_FIELD and item.parameter_field == "expr":
            self._value_delegate.show_expr_popup(index)

    def on_controller_selection_changed(
        self,
        spectrum_id: str | None,
        _region_id: str | None,
        _component_id: str | None,
    ) -> None:
        """
        Rebuild when the spectrum changes; otherwise only sync row selection.

        Parameters
        ----------
        spectrum_id : str or None
            Newly selected spectrum.
        _region_id : str or None
            Newly selected region (unused beyond sync).
        _component_id : str or None
            Newly selected component (unused beyond sync).
        """
        if spectrum_id != self._last_spectrum_id:
            self.refresh()
            return
        if self._updating_from_view:
            return
        self.sync_selection_from_controller()

    def sync_selection_from_controller(self) -> None:
        """Select the row matching the controller's region/component selection."""
        component_id = self._controller.selected_component_id
        region_id = self._controller.selected_region_id
        target = self._find_index_for_selection(component_id=component_id, region_id=region_id)
        self._syncing_selection = True
        try:
            sel = self.selectionModel()
            if target is None or not target.isValid():
                sel.clearSelection()
                self._sync_parameter_slider_editor()
                return
            parent = target.parent()
            while parent.isValid():
                self.setExpanded(parent, True)
                parent = parent.parent()
            sel.select(
                target,
                sel.SelectionFlag.ClearAndSelect | sel.SelectionFlag.Rows,
            )
            self.setCurrentIndex(target)
            self.scrollTo(target)
        finally:
            self._syncing_selection = False
        self._sync_parameter_slider_editor()

    def _find_index_for_selection(
        self,
        *,
        component_id: str | None,
        region_id: str | None,
    ) -> QModelIndex | None:
        """Return the model index for the given component or region, if present."""

        def walk(parent: QModelIndex) -> QModelIndex | None:
            for row in range(self._model.rowCount(parent)):
                idx = self._model.index(row, 0, parent)
                raw = idx.internalPointer()
                if isinstance(raw, PropertyItem):
                    if (
                        component_id is not None
                        and raw.kind == ItemKind.COMPONENT
                        and raw.component_id == component_id
                    ):
                        return idx
                    if (
                        component_id is None
                        and region_id is not None
                        and raw.kind == ItemKind.REGION
                        and raw.region_id == region_id
                    ):
                        return idx
                found = walk(idx)
                if found is not None:
                    return found
            return None

        return walk(QModelIndex())

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
        if item.kind == ItemKind.PARAMETER_FIELD:
            return (
                "param_field",
                item.component_id,
                item.parameter_name,
                item.parameter_field,
            )
        if item.kind == ItemKind.ACTION_ROW:
            return ("action", item.action, item.region_id)
        return ("other", item.kind, item.name)

    def _stable_key_for_current_item(self) -> tuple[Any, ...] | None:
        """Return a stable key for the current selection, if any."""
        current = self.selectionModel().currentIndex()
        if not current.isValid():
            return None
        item = current.internalPointer()
        if not isinstance(item, PropertyItem):
            return None
        if item.kind in {
            ItemKind.PARAMETER_ROW,
            ItemKind.PARAMETER_FIELD,
            ItemKind.REGION_SLICE,
            ItemKind.COMPONENT,
            ItemKind.COMPONENT_MODEL,
            ItemKind.REGION,
            ItemKind.ACTION_ROW,
        }:
            return self._stable_key_for_item(item)
        return None

    def _select_by_stable_key(self, key: tuple[Any, ...], column: int = 0) -> bool:
        """
        Select the row matching ``key`` and expand ancestors.

        Returns
        -------
        bool
            True when a matching row was found and selected.
        """
        found = self._find_index_by_stable_key(key)
        if found is None:
            return False
        # Expand ancestors so nested parameter fields are visible.
        parent = found.parent()
        while parent.isValid():
            self.expand(parent)
            parent = parent.parent()
        col = max(0, min(column, self._model.columnCount() - 1))
        target = self._model.index(found.row(), col, found.parent())
        self._syncing_selection = True
        try:
            self.setCurrentIndex(target)
            self.selectionModel().select(
                target,
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows,
            )
        finally:
            self._syncing_selection = False
        return True

    def _find_index_by_stable_key(self, key: tuple[Any, ...]) -> QModelIndex | None:
        """Return column-0 index for the item with the given stable key."""

        def walk(parent: QModelIndex) -> QModelIndex | None:
            for row in range(self._model.rowCount(parent)):
                idx = self._model.index(row, 0, parent)
                raw = idx.internalPointer()
                if isinstance(raw, PropertyItem) and self._stable_key_for_item(raw) == key:
                    return idx
                nested = walk(idx)
                if nested is not None:
                    return nested
            return None

        return walk(QModelIndex())

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
        Update controller region/component selection from the properties view.

        Parameters
        ----------
        selected : Any
            Newly selected indexes.
        _deselected : Any
            No longer selected indexes (unused).
        """
        del _deselected
        if self._syncing_selection:
            return

        indexes = selected.indexes()
        if not indexes:
            # Intermediate clears (cell editor handoff / model reset) must not
            # collapse the open parameter detail.
            return

        index = indexes[0]
        item = index.internalPointer()
        if not isinstance(item, PropertyItem):
            return

        region_id: str | None = None
        component_id: str | None = None
        cursor: PropertyItem | None = item
        while cursor is not None:
            if component_id is None and cursor.component_id is not None:
                if cursor.kind in {
                    ItemKind.COMPONENT,
                    ItemKind.COMPONENT_MODEL,
                    ItemKind.PARAMETER_ROW,
                    ItemKind.PARAMETER_FIELD,
                }:
                    component_id = cursor.component_id
            if region_id is None and cursor.region_id is not None:
                region_id = cursor.region_id
            cursor = cursor.parent

        self._updating_from_view = True
        try:
            self._controller.set_selection(
                self._controller.selected_spectrum_id,
                region_id,
                component_id,
            )
        finally:
            self._updating_from_view = False
        self._sync_parameter_slider_editor()

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

    def _copy_value_at(self, index: QModelIndex) -> None:
        """Copy the value / lower / upper / expr cell text to the clipboard."""
        item = index.internalPointer() if index.isValid() else None
        if not isinstance(item, PropertyItem) or not _is_copyable_value_item(item):
            return
        QApplication.clipboard().setText(_clipboard_text_for_item(item))
        self._value_delegate.flash_copy_success(index)

    def _delete_selected_component(self) -> None:
        """Remove the selected component and refresh."""
        cid = self._selected_component_id()
        if cid:
            self._discard_parameter_slider_editor()
            self._controller.full_remove_object(cid)
            self.refresh()

    def _on_action_row_requested(self, index: QModelIndex) -> None:
        """Run the mutation associated with a clicked ``ACTION_ROW``."""
        item = index.internalPointer() if index.isValid() else None
        if not isinstance(item, PropertyItem) or item.kind != ItemKind.ACTION_ROW:
            return
        params = self._controller.get_app_parameters()
        if item.action == "add_region":
            spectrum_id = self._controller.selected_spectrum_id
            if spectrum_id is None:
                return
            self._controller.create_region(spectrum_id)
            return
        if item.action == "add_peak":
            if item.region_id is None:
                return
            self._controller.create_peak(item.region_id, params.default_peak_model, parameters=None)
            return
        if item.action == "add_background":
            if item.region_id is None:
                return
            self._controller.create_background(
                item.region_id, params.default_background_model, parameters=None
            )

    def _on_row_optimize_requested(self, index: QModelIndex) -> None:
        """Optimize the single region for the clicked region row."""
        item = index.internalPointer() if index.isValid() else None
        if not isinstance(item, PropertyItem) or item.kind != ItemKind.REGION:
            return
        if item.region_id is None:
            return
        confirm_and_optimize(
            self,
            self._controller,
            region_ids=[item.region_id],
        )

    def _on_row_delete_requested(self, index: QModelIndex) -> None:
        """Delete the region or component for the clicked row."""
        item = index.internalPointer() if index.isValid() else None
        if not isinstance(item, PropertyItem):
            return
        # Drop any open slice/parameter preview first so refresh cannot commit
        # against an object that is about to disappear.
        self._discard_parameter_slider_editor()
        if item.kind == ItemKind.REGION and item.region_id is not None:
            region_id = item.region_id
            spectrum_id = self._controller.selected_spectrum_id
            if self._controller.selected_region_id == region_id:
                self._controller.set_selection(spectrum_id, None)
            self._controller.full_remove_object(region_id)
            return
        if item.kind == ItemKind.COMPONENT and item.component_id is not None:
            self._controller.full_remove_object(item.component_id)

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

        menu.popup(self.viewport().mapToGlobal(pos))
