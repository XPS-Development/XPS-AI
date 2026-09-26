"""Tests for copy-decomposition link tree tri-state checkboxes."""

from __future__ import annotations

import sys
from typing import cast
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QEvent, QModelIndex, QPointF, QRect, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem

from core.dto import ComponentDTO, ParameterDTO
from ui.component_colors import color_for_component
from ui.dialogs.copy_decomposition_dialog import LinkCheckboxDelegate, LinkTreeModel
from ui.trees.name_id_delegate import ComponentColorRole, ObjectIdPrefixRole


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for QAbstractItemModel tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast(QApplication, app)


def _param(name: str, value: float = 1.0) -> ParameterDTO:
    return ParameterDTO(name=name, value=value, lower=0.0, upper=10.0, vary=True, expr=None)


def _peak_dto(component_id: str, name: str) -> ComponentDTO:
    model = MagicMock()
    model.name = "pseudo-voigt"
    return ComponentDTO(
        id_=component_id,
        parent_id="r1",
        normalized=False,
        parameters={
            "amp": _param("amp"),
            "cen": _param("cen", 0.0),
            "sig": _param("sig"),
            "frac": _param("frac", 0.5),
        },
        model=model,
        kind="peak",
        name=name,
    )


@pytest.fixture
def link_model(qapp: QApplication) -> LinkTreeModel:
    """LinkTreeModel with one region and two peaks."""
    del qapp
    controller = MagicMock()
    query = controller.query
    query.get_regions_ids.return_value = ("r1abcdxxxx",)
    query.get_background_id.return_value = None
    query.get_peaks_ids.return_value = ("p1abcd1111", "p2abcd2222")
    query.get_component_dto.side_effect = lambda cid, normalized=False: _peak_dto(
        cid, "Peak A" if cid.startswith("p1") else "Peak B"
    )
    return LinkTreeModel(controller, source_spectrum_id="s1")


def _link_index(model: LinkTreeModel, *rows: int) -> QModelIndex:
    """Return column-1 (Link) index for a path of column-0 rows."""
    parent = QModelIndex()
    name_index = QModelIndex()
    for row in rows:
        name_index = model.index(row, 0, parent)
        parent = name_index
    return name_index.sibling(name_index.row(), 1)


def test_link_tree_defaults_unchecked(link_model: LinkTreeModel) -> None:
    """All parameter link flags start unchecked."""
    flags = link_model.link_flags()
    assert flags
    assert all(linked is False for linked in flags.values())


def test_link_tree_region_cascade_and_partial(link_model: LinkTreeModel) -> None:
    """Checking a region cascades; partial child selection yields PartiallyChecked."""
    region_link = _link_index(link_model, 0)
    assert link_model.data(region_link, Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Unchecked

    assert link_model.setData(region_link, Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
    assert link_model.data(region_link, Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
    assert all(link_model.link_flags().values())

    param_amp = _link_index(link_model, 0, 0, 0)
    assert link_model.setData(param_amp, Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    assert (
        link_model.data(region_link, Qt.ItemDataRole.CheckStateRole)
        == Qt.CheckState.PartiallyChecked
    )
    assert link_model.link_flags()[("p1abcd1111", "amp")] is False


def test_link_tree_clear_links(link_model: LinkTreeModel) -> None:
    """clear_links resets every checkbox."""
    region_link = _link_index(link_model, 0)
    link_model.setData(region_link, Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
    link_model.clear_links()
    assert all(linked is False for linked in link_model.link_flags().values())


def _click_link(
    delegate: LinkCheckboxDelegate,
    model: LinkTreeModel,
    index: QModelIndex,
    event_type: QEvent.Type,
) -> bool:
    """Send one left-button event to the Link checkbox delegate."""
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 48, 24)
    event = QMouseEvent(
        event_type,
        QPointF(10, 10),
        QPointF(10, 10),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    return delegate.editorEvent(event, model, option, index)


def test_link_checkbox_double_click_toggles_once_per_click(link_model: LinkTreeModel) -> None:
    """A fast second click must not toggle again on the double-click event."""
    delegate = LinkCheckboxDelegate()
    index = _link_index(link_model, 0, 0, 0)
    check = Qt.ItemDataRole.CheckStateRole

    assert _click_link(delegate, link_model, index, QEvent.Type.MouseButtonPress)
    assert link_model.data(index, check) == Qt.CheckState.Unchecked
    assert _click_link(delegate, link_model, index, QEvent.Type.MouseButtonRelease)
    assert link_model.data(index, check) == Qt.CheckState.Checked

    assert _click_link(delegate, link_model, index, QEvent.Type.MouseButtonDblClick)
    assert link_model.data(index, check) == Qt.CheckState.Checked
    assert _click_link(delegate, link_model, index, QEvent.Type.MouseButtonRelease)
    assert link_model.data(index, check) == Qt.CheckState.Unchecked


def test_link_tree_shows_short_id_and_component_color(link_model: LinkTreeModel) -> None:
    """Region/component rows expose short id prefix and peak color swatches."""
    region = link_model.index(0, 0, QModelIndex())
    assert link_model.data(region, ObjectIdPrefixRole) == "r1abc"
    component = link_model.index(0, 0, region)
    assert link_model.data(component, ObjectIdPrefixRole) == "p1abc"
    assert link_model.data(component, ComponentColorRole) == color_for_component(
        component_id="p1abcd1111"
    )
    assert (
        link_model.headerData(1, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) == "Link"
    )
