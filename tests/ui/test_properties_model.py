"""Tests for :class:`ui.properties.PropertiesModel`."""

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QModelIndex, Qt

from ui.properties import ItemKind, PropertiesModel, PropertyItem


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
    """COMPONENT rows expose truncated id and a stable color string."""
    from ui.name_id_delegate import ComponentColorRole, ObjectIdPrefixRole, ObjectIdRole

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
    assert isinstance(model.data(index, ComponentColorRole), str)
