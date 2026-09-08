"""Tests for :class:`ui.spectrum_tree.SpectrumTreeModel`."""

import sys
from typing import cast

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from tests.conftest import seed_hierarchy_metadata

from ui.controller import ControllerWrapper
from ui.spectrum_tree import SpectrumTreeModel


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
    """show_spectrum_id_in_tree appends a short id suffix to spectrum labels."""
    del qapp
    controller = ControllerWrapper(collection=hierarchy_collection)
    controller.orchestrator._params.show_spectrum_id_in_tree = True
    seed_hierarchy_metadata(controller.orchestrator.ctx.metadata)
    model = SpectrumTreeModel(controller)

    file_index = model.index(0, 0)
    group_a_index = model.index(0, 0, file_index)
    spectrum_labels = _child_labels(model, group_a_index)

    assert any(label.endswith(" s1") for label in spectrum_labels)
    assert any(label.endswith(" s2") for label in spectrum_labels)
