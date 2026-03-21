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
    """Model exposes six columns with expected header labels."""
    model = PropertiesModel(mock_controller)
    model.refresh()
    assert model.columnCount() == 6
    assert model.headerData(0, Qt.Orientation.Horizontal, Qt.DisplayRole) == "Name"
    assert model.headerData(1, Qt.Orientation.Horizontal, Qt.DisplayRole) == "Value"
    assert model.headerData(5, Qt.Orientation.Horizontal, Qt.DisplayRole) == "Expr"


def test_parameter_row_data_maps_columns(mock_controller: MagicMock) -> None:
    """PARAMETER_ROW serves value/lower/upper/vary/expr across columns 1–5."""
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
        param_expr=None,
    )
    model._root_item.append_child(row)
    model.beginResetModel()
    model.endResetModel()

    def idx(c: int):
        return model.index(0, c, QModelIndex())

    assert model.data(idx(0), Qt.DisplayRole) == "amplitude"
    assert model.data(idx(1), Qt.DisplayRole) == "1.50"
    assert model.data(idx(2), Qt.DisplayRole) == "0.00"
    assert model.data(idx(3), Qt.DisplayRole) == "10.00"
    assert model.data(idx(4), Qt.CheckStateRole) == Qt.Checked
    assert model.data(idx(5), Qt.DisplayRole) == ""
