from dataclasses import dataclass, field
from typing import Any, Optional

from PySide6.QtCore import QAbstractItemModel, QModelIndex, QPoint, Qt
from PySide6.QtWidgets import QMenu, QTreeView, QWidget

from app.command.changes import ParameterField

from .controller import ControllerWrapper


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
    value : str, optional
        Text shown in the second column for leaf nodes.
    parent : PropertyItem or None, optional
        Parent item in the tree.
    """

    name: str
    value: str = ""
    parent: Optional["PropertyItem"] = None
    children: list["PropertyItem"] = field(default_factory=list)
    kind: str = "group"
    region_id: Optional[str] = None
    component_id: Optional[str] = None
    parameter_name: Optional[str] = None
    parameter_field: Optional[ParameterField] = None

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
        return 0

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
        self._root_item = PropertyItem(name="Root")

    # ------------------------------------------------------------------
    # Required model API
    # ------------------------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if not parent.isValid():
            item = self._root_item
        else:
            item = parent.internalPointer()
        if not isinstance(item, PropertyItem):
            return 0
        return len(item.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        del parent
        return 2

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:  # noqa: N802
        if row < 0 or column < 0:
            return QModelIndex()

        if not parent.isValid():
            parent_item = self._root_item
        else:
            parent_item = parent.internalPointer()

        if not isinstance(parent_item, PropertyItem):
            return QModelIndex()

        child_item = parent_item.child(row)
        if child_item is None:
            return QModelIndex()
        return self.createIndex(row, column, child_item)

    def parent(self, index: QModelIndex) -> QModelIndex:  # noqa: N802
        if not index.isValid():
            return QModelIndex()

        item = index.internalPointer()
        if not isinstance(item, PropertyItem):
            return QModelIndex()

        parent_item = item.parent
        if parent_item is None or parent_item is self._root_item:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if not index.isValid():
            return None

        item = index.internalPointer()
        if not isinstance(item, PropertyItem):
            return None

        if role in (Qt.DisplayRole, Qt.EditRole):
            if index.column() == 0:
                return item.name
            if index.column() == 1:
                return item.value

        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:  # noqa: N802
        if not index.isValid():
            return Qt.NoItemFlags

        base_flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        item = index.internalPointer()
        if not isinstance(item, PropertyItem):
            return base_flags

        if index.column() != 1:
            return base_flags

        if item.kind in {"region_slice", "parameter_field"}:
            return base_flags | Qt.ItemIsEditable

        return base_flags

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:  # noqa: N802
        if role != Qt.EditRole or not index.isValid():
            return False

        item = index.internalPointer()
        if not isinstance(item, PropertyItem) or index.column() != 1:
            return False

        if item.kind == "region_slice" and item.region_id is not None:
            try:
                new_bound = int(value)
            except (TypeError, ValueError):
                return False

            region_service = self._controller.ctx.region
            start, stop = region_service.get_slice(item.region_id, mode="index")
            start = start or 0
            stop = stop or 0

            if item.name == "start":
                start = new_bound
            elif item.name == "stop":
                stop = new_bound
            else:
                return False

            try:
                self._controller.update_region_slice(item.region_id, start, stop)
            except Exception:  # noqa: BLE001
                return False

            item.value = str(new_bound)
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True

        if (
            item.kind == "parameter_field"
            and item.component_id is not None
            and item.parameter_name is not None
        ):
            field = item.parameter_field
            if field is None:
                return False

            try:
                coerced: str | bool | float
                text = str(value)
                if field in {"value", "lower", "upper"}:
                    coerced = float(text)
                elif field == "vary":
                    lowered = text.strip().lower()
                    if lowered in {"1", "true", "yes", "on"}:
                        coerced = True
                    elif lowered in {"0", "false", "no", "off"}:
                        coerced = False
                    else:
                        return False
                elif field in {"name", "expr"}:
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

            item.value = str(value)
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

        query = self._controller.ctx.query
        region_service = self._controller.ctx.region
        dto_service = self._controller.dto_service

        region_ids = query.get_regions(spectrum_id)
        if not region_ids:
            self._root_item.append_child(PropertyItem(name="No regions", parent=self._root_item))
            self.endResetModel()
            return

        for index, region_id in enumerate(region_ids, start=1):
            region_label = f"Region {index}"
            region_item = PropertyItem(
                name=region_label,
                parent=self._root_item,
                kind="region",
                region_id=region_id,
            )
            self._root_item.append_child(region_item)

            start_val, stop_val = region_service.get_slice(region_id, mode="index")
            region_item.append_child(
                PropertyItem(
                    name="start",
                    value=str(start_val),
                    parent=region_item,
                    kind="region_slice",
                    region_id=region_id,
                )
            )
            region_item.append_child(
                PropertyItem(
                    name="stop",
                    value=str(stop_val),
                    parent=region_item,
                    kind="region_slice",
                    region_id=region_id,
                )
            )

            background_id = query.get_background(region_id)
            peaks_ids = list(query.get_peaks(region_id))

            if background_id is not None:
                background_item = PropertyItem(
                    name="Background",
                    parent=region_item,
                    kind="component",
                    region_id=region_id,
                    component_id=background_id,
                )
                region_item.append_child(background_item)
                background_dto = dto_service.get_component(background_id)
                model_name = type(background_dto.model).__name__
                background_item.append_child(
                    PropertyItem(
                        name="model",
                        value=str(model_name),
                        parent=background_item,
                        kind="group",
                        component_id=background_id,
                    )
                )
                params = background_dto.parameters
                for param_name, param_dto in params.items():
                    param_group = PropertyItem(
                        name=str(param_name),
                        parent=background_item,
                        kind="parameter",
                        component_id=background_id,
                        parameter_name=str(param_name),
                    )
                    background_item.append_child(param_group)
                    for field_name in ("name", "value", "lower", "upper", "vary", "expr"):
                        field_value = getattr(param_dto, field_name)
                        param_group.append_child(
                            PropertyItem(
                                name=field_name,
                                value=str(field_value),
                                parent=param_group,
                                kind="parameter_field",
                                component_id=background_id,
                                parameter_name=str(param_name),
                                parameter_field=field_name,  # type: ignore[arg-type]
                            )
                        )

            for peak_index, peak_id in enumerate(peaks_ids, start=1):
                peak_item = PropertyItem(
                    name=f"Peak {peak_index}",
                    parent=region_item,
                    kind="component",
                    region_id=region_id,
                    component_id=peak_id,
                )
                region_item.append_child(peak_item)
                peak_dto = dto_service.get_component(peak_id)
                model_name = type(peak_dto.model).__name__
                peak_item.append_child(
                    PropertyItem(
                        name="model",
                        value=str(model_name),
                        parent=peak_item,
                        kind="group",
                        component_id=peak_id,
                    )
                )
                params = peak_dto.parameters
                for param_name, param_dto in params.items():
                    param_group = PropertyItem(
                        name=str(param_name),
                        parent=peak_item,
                        kind="parameter",
                        component_id=peak_id,
                        parameter_name=str(param_name),
                    )
                    peak_item.append_child(param_group)
                    for field_name in ("name", "value", "lower", "upper", "vary", "expr"):
                        field_value = getattr(param_dto, field_name)
                        param_group.append_child(
                            PropertyItem(
                                name=field_name,
                                value=str(field_value),
                                parent=param_group,
                                kind="parameter_field",
                                component_id=peak_id,
                                parameter_name=str(param_name),
                                parameter_field=field_name,  # type: ignore[arg-type]
                            )
                        )

        self.endResetModel()


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
        self.setHeaderHidden(False)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_custom_context_menu)
        self.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self._model.refresh()
        self.expandAll()

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

        menu = QMenu(self)

        if spectrum_id is not None:
            menu.addAction("Add region", self._create_full_region_for_spectrum)

        if region_id is not None:
            menu.addAction("Add peak", lambda rid=region_id: self._create_peak_for_region(rid))
            menu.addAction("Set background", lambda rid=region_id: self._set_background_for_region(rid))

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

        dto_service = self._controller.dto_service
        try:
            spectrum_dto = dto_service.get_spectrum(spectrum_id, normalized=False)
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
