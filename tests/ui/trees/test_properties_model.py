"""Tests for :class:`ui.trees.properties.PropertiesModel`."""

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QModelIndex, Qt

from ui.trees.properties import ItemKind, PropertiesModel, PropertyItem, _is_copyable_value_item


@pytest.fixture
def mock_controller() -> MagicMock:
    """Minimal controller stub for :meth:`PropertiesModel.refresh`."""
    c = MagicMock()
    c.selected_spectrum_id = None
    params = MagicMock()
    params.region_slice_display_mode = "index"
    params.show_id_in_properties_tree = True
    c.get_app_parameters.return_value = params
    return c


def test_properties_model_column_count_and_headers(mock_controller: MagicMock) -> None:
    """Model exposes two columns: Name and Value."""
    model = PropertiesModel(mock_controller)
    model.refresh()
    assert model.columnCount() == 2
    assert model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) == "Name"
    assert model.headerData(1, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) == "Value"


def test_parameter_row_data_maps_columns(mock_controller: MagicMock) -> None:
    """PARAMETER_ROW serves value; nested fields hold bounds / vary / expr."""
    del mock_controller  # unused; build a minimal tree without refresh()
    model = PropertiesModel(MagicMock(selected_spectrum_id=None))
    model._root_item.children.clear()
    row = PropertyItem(
        name="amplitude",
        value=1.5,
        parent=model._root_item,
        kind=ItemKind.PARAMETER_ROW,
        component_id="c1",
        parameter_name="amplitude",
        param_lower=0.0,
        param_upper=10.0,
        param_vary=True,
        param_expr="pOther",
    )
    model._root_item.append_child(row)
    for field_name, field_value in (
        ("lower", 0.0),
        ("upper", 10.0),
        ("vary", True),
        ("expr", "pOther"),
    ):
        row.append_child(
            PropertyItem(
                name=field_name,
                value=field_value,
                parent=row,
                kind=ItemKind.PARAMETER_FIELD,
                component_id="c1",
                parameter_name="amplitude",
                parameter_field=field_name,  # type: ignore[arg-type]
            )
        )
    model.beginResetModel()
    model.endResetModel()

    def idx(c: int):
        return model.index(0, c, QModelIndex())

    assert model.data(idx(0), Qt.ItemDataRole.DisplayRole) == "amplitude  ƒ"
    assert model.data(idx(1), Qt.ItemDataRole.DisplayRole) == "1.50"
    assert model.data(idx(0), Qt.ItemDataRole.ToolTipRole) is None

    param = model.index(0, 0, QModelIndex())
    lower = model.index(0, 1, param)
    vary = model.index(2, 1, param)
    expr = model.index(3, 1, param)
    assert model.data(lower, Qt.ItemDataRole.DisplayRole) == "0.00"
    assert model.data(vary, Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
    assert model.data(expr, Qt.ItemDataRole.DisplayRole) == "pOther"


def test_refresh_adds_action_rows_for_empty_spectrum(mock_controller: MagicMock) -> None:
    """Spectrum with no regions still shows a root Add region action row."""
    mock_controller.selected_spectrum_id = "s1"
    query = MagicMock()
    query.get_regions_ids.return_value = []
    mock_controller.query = query

    model = PropertiesModel(mock_controller)
    model.refresh()

    assert model.rowCount(QModelIndex()) == 1
    root_action = model.index(0, 0, QModelIndex())
    item = root_action.internalPointer()
    assert isinstance(item, PropertyItem)
    assert item.kind == ItemKind.ACTION_ROW
    assert item.action == "add_region"
    assert model.data(root_action, Qt.ItemDataRole.DisplayRole) == "Add region"


def test_refresh_region_actions_when_empty(mock_controller: MagicMock) -> None:
    """Empty region shows Add background and Add peak; Add region stays at root."""
    mock_controller.selected_spectrum_id = "s1"
    query = MagicMock()
    query.get_regions_ids.return_value = ["r1"]
    query.get_region_slice.return_value = (0, 10)
    query.get_background_id.return_value = None
    query.get_peaks_ids.return_value = []
    query.get_region_dto.return_value = MagicMock()
    mock_controller.query = query
    mock_controller._region_soft_context = MagicMock(return_value=(0.0, 10.0, 1.0))

    model = PropertiesModel(mock_controller)
    # Avoid soft-range lookup complexity: stub helper used by refresh.
    model._region_soft_context = MagicMock(return_value=(0.0, 10.0, 1.0))  # type: ignore[method-assign]
    model._slice_soft_range = MagicMock(return_value=(0.0, 10.0))  # type: ignore[method-assign]
    model.refresh()

    region = model.index(0, 0, QModelIndex())
    add_region = model.index(1, 0, QModelIndex())
    assert add_region.internalPointer().action == "add_region"  # type: ignore[union-attr]

    kinds = []
    actions = []
    for row in range(model.rowCount(region)):
        child = model.index(row, 0, region).internalPointer()
        assert isinstance(child, PropertyItem)
        kinds.append(child.kind)
        actions.append(child.action)

    assert ItemKind.ACTION_ROW in kinds
    assert "add_background" in actions
    assert "add_peak" in actions
    assert actions.index("add_background") < actions.index("add_peak")


def test_component_row_exposes_gray_id_and_color(mock_controller: MagicMock) -> None:
    """COMPONENT rows expose truncated id and a color tied to the peak id."""
    from ui.component_colors import color_for_component
    from ui.trees.name_id_delegate import ComponentColorRole, ObjectIdPrefixRole, ObjectIdRole

    mock_controller.get_app_parameters.return_value.show_id_in_properties_tree = True
    model = PropertiesModel(mock_controller)
    model._root_item.children.clear()
    peak = PropertyItem(
        name="Peak 1",
        parent=model._root_item,
        kind=ItemKind.COMPONENT,
        region_id="r1",
        component_id="pabcdef123",
        component_kind="peak",
        object_id="pabcdef123",
    )
    model._root_item.append_child(peak)
    model.beginResetModel()
    model.endResetModel()
    index = model.index(0, 0, QModelIndex())
    assert model.data(index, ObjectIdRole) == "pabcdef123"
    assert model.data(index, ObjectIdPrefixRole) == "pabcd"
    assert model.data(index, ComponentColorRole) == color_for_component(component_id="pabcdef123")


def test_region_and_peak_area_is_read_only_and_copyable(mock_controller: MagicMock) -> None:
    """Fitted area sits opposite the name: shown and copyable, not editable."""
    del mock_controller
    model = PropertiesModel(MagicMock(selected_spectrum_id=None))
    model._root_item.children.clear()
    region = PropertyItem(
        name="Region 1",
        value=3.5,
        parent=model._root_item,
        kind=ItemKind.REGION,
        region_id="r1",
    )
    peak = PropertyItem(
        name="Peak 1",
        value=3.5,
        parent=region,
        kind=ItemKind.COMPONENT,
        component_kind="peak",
        component_id="p1",
    )
    background = PropertyItem(
        name="Background",
        parent=region,
        kind=ItemKind.COMPONENT,
        component_kind="background",
        component_id="b1",
    )
    region.append_child(peak)
    region.append_child(background)
    model._root_item.append_child(region)
    model.beginResetModel()
    model.endResetModel()

    region_value = model.index(0, 1, QModelIndex())
    peak_value = model.index(0, 1, model.index(0, 0, QModelIndex()))
    background_value = model.index(1, 1, model.index(0, 0, QModelIndex()))

    assert model.data(region_value, Qt.ItemDataRole.DisplayRole) == "s = 3.50"
    assert model.data(peak_value, Qt.ItemDataRole.DisplayRole) == "s = 3.50"
    assert model.data(background_value, Qt.ItemDataRole.DisplayRole) is None
    assert model.data(region_value, Qt.ItemDataRole.ToolTipRole) is None

    editable = Qt.ItemFlag.ItemIsEditable
    assert editable not in model.flags(region_value)
    assert editable not in model.flags(peak_value)
    region_item = region_value.internalPointer()
    peak_item = peak_value.internalPointer()
    background_item = background_value.internalPointer()
    assert isinstance(region_item, PropertyItem)
    assert isinstance(peak_item, PropertyItem)
    assert isinstance(background_item, PropertyItem)
    assert _is_copyable_value_item(region_item) is True
    assert _is_copyable_value_item(peak_item) is True
    assert _is_copyable_value_item(background_item) is False
