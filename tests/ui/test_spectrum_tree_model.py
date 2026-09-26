"""Tests for :class:`ui.spectrum_tree.SpectrumTreeModel`."""

import sys
from typing import cast

import pytest
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import QApplication
from tests.conftest import seed_hierarchy_metadata

from ui.controller import ControllerWrapper
from ui.spectrum_tree import SpectrumTreeModel
from ui.spectrum_tree_panel import SpectrumTreePanel


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for QAbstractItemModel tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast(QApplication, app)


def _child_labels(model: SpectrumTreeModel, parent_index) -> list[str]:
    return [
        model.data(model.index(row, 0, parent_index), Qt.ItemDataRole.DisplayRole)
        for row in range(model.rowCount(parent_index))
    ]


def test_spectrum_tree_model_builds_file_group_hierarchy(
    qapp: QApplication,
    hierarchy_collection,
) -> None:
    """refresh() groups spectra by metadata file and group."""
    del qapp
    controller = ControllerWrapper(collection=hierarchy_collection)
    controller.orchestrator._params.show_spectrum_id_in_tree = False
    seed_hierarchy_metadata(controller.orchestrator.ctx.metadata)
    model = SpectrumTreeModel(controller)

    assert model.rowCount() == 1
    file_index = model.index(0, 0)
    assert model.data(file_index, Qt.ItemDataRole.DisplayRole) == "file-a"
    assert model.rowCount(file_index) == 2

    group_labels = _child_labels(model, file_index)
    assert group_labels == ["group-a", "group-b"]

    group_a_index = model.index(0, 0, file_index)
    assert model.rowCount(group_a_index) == 2
    assert sorted(_child_labels(model, group_a_index)) == ["spec-1", "spec-2"]

    group_b_index = model.index(1, 0, file_index)
    assert model.rowCount(group_b_index) == 1
    assert _child_labels(model, group_b_index) == ["spec-3"]


def test_spectrum_tree_model_shows_spectrum_id_suffix_when_enabled(
    qapp: QApplication,
    hierarchy_collection,
) -> None:
    """show_spectrum_id_in_tree exposes a short id via ObjectIdPrefixRole."""
    from ui.name_id_delegate import ObjectIdPrefixRole

    del qapp
    controller = ControllerWrapper(collection=hierarchy_collection)
    controller.orchestrator._params.show_spectrum_id_in_tree = True
    seed_hierarchy_metadata(controller.orchestrator.ctx.metadata)
    model = SpectrumTreeModel(controller)

    file_index = model.index(0, 0)
    group_a_index = model.index(0, 0, file_index)
    prefixes = [
        model.data(model.index(row, 0, group_a_index), ObjectIdPrefixRole)
        for row in range(model.rowCount(group_a_index))
    ]
    labels = _child_labels(model, group_a_index)

    assert sorted(labels) == ["spec-1", "spec-2"]
    assert "s1" in prefixes
    assert "s2" in prefixes


def _spectrum_row(panel: SpectrumTreePanel, spectrum_id: str) -> tuple[QModelIndex, int]:
    model = panel.model

    def walk(parent: QModelIndex) -> tuple[QModelIndex, int] | None:
        for row in range(model.rowCount(parent)):
            index = model.index(row, 0, parent)
            item = model.item_from_index(index)
            if item is not None and item.spectrum_id == spectrum_id:
                return parent, row
            found = walk(index)
            if found is not None:
                return found
        return None

    found = walk(QModelIndex())
    assert found is not None
    return found


def test_spectrum_tree_search_matches_spectrum_id(
    qapp: QApplication,
    hierarchy_collection,
) -> None:
    """Typing a spectrum id keeps that row and hides the others."""
    del qapp
    controller = ControllerWrapper(collection=hierarchy_collection)
    seed_hierarchy_metadata(controller.orchestrator.ctx.metadata)
    panel = SpectrumTreePanel(controller)
    panel._search_edit.setText("s3")

    hidden_parent, hidden_row = _spectrum_row(panel, "s1")
    shown_parent, shown_row = _spectrum_row(panel, "s3")
    assert panel.tree.isRowHidden(hidden_row, hidden_parent)
    assert not panel.tree.isRowHidden(shown_row, shown_parent)
    shown_by_id = panel.model.item_from_index(panel.model.index(shown_row, 0, shown_parent))
    assert shown_by_id is not None
    assert "s3" in shown_by_id.search_text
    assert panel._scroll_target.isValid()
    shown_item = panel.model.item_from_index(panel._scroll_target)
    assert shown_item is not None
    assert shown_item.spectrum_id == "s3"
    panel.deleteLater()
    QApplication.processEvents()


def test_spectrum_tree_search_matches_peak_id(
    qapp: QApplication,
    hierarchy_collection,
) -> None:
    """A peak or region id finds the spectrum that owns it and hides the rest."""
    del qapp
    controller = ControllerWrapper(collection=hierarchy_collection)
    seed_hierarchy_metadata(controller.orchestrator.ctx.metadata)
    controller.create_region("s1", start=0, stop=10, region_id="r-owned", mode="index")
    controller.create_peak(
        "r-owned",
        "pseudo-voigt",
        parameters={"amp": 1.0, "cen": 0.0, "sig": 1.0, "frac": 0.5},
        peak_id="p-owned",
    )
    panel = SpectrumTreePanel(controller)
    panel._search_edit.setText("p-owned")

    hidden_parent, hidden_row = _spectrum_row(panel, "s2")
    shown_parent, shown_row = _spectrum_row(panel, "s1")
    assert panel.tree.isRowHidden(hidden_row, hidden_parent)
    assert not panel.tree.isRowHidden(shown_row, shown_parent)

    panel._search_edit.setText("r-owned")
    hidden_parent, hidden_row = _spectrum_row(panel, "s3")
    shown_parent, shown_row = _spectrum_row(panel, "s1")
    assert panel.tree.isRowHidden(hidden_row, hidden_parent)
    assert not panel.tree.isRowHidden(shown_row, shown_parent)
    panel.deleteLater()
    QApplication.processEvents()
