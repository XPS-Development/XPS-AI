"""Tests for the expression editor popup."""

from __future__ import annotations

import sys
from typing import cast
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import QApplication

from ui.component_colors import color_for_component
from ui.dialogs.expr_editor_popup import ExprEditorPopup, ExprPickerModel
from ui.trees.name_id_delegate import ComponentColorRole, ObjectIdPrefixRole


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast(QApplication, app)


def _build_controller() -> MagicMock:
    """Minimal controller stub with one spectrum / region / peak / background."""
    c = MagicMock()
    c.selected_spectrum_id = "s0d8eabc"
    query = c.query
    query.get_all_spectra_ids.return_value = ("s0d8eabc",)
    meta = MagicMock()
    meta.file = "/data/sample.vms"
    meta.group = "Group A"
    meta.name = "Spec 1"
    query.get_metadata.return_value = meta
    query.get_regions_ids.return_value = ("r3113xyz",)
    query.get_background_id.return_value = "b9f01abc"
    query.get_peaks_ids.return_value = ("pabcd111", "p12ef222")
    query.get_spectrum_structure_status.return_value = "peaks"

    def _component_dto(cid: str) -> MagicMock:
        dto = MagicMock()
        if cid.startswith("b"):
            dto.name = "Background"
        elif cid.startswith("pabcd"):
            dto.name = "Peak 1"
        else:
            dto.name = "Peak 2"
        return dto

    query.get_component_dto.side_effect = _component_dto
    return c


def test_expr_picker_model_exposes_status_and_ids(qapp: QApplication) -> None:
    """Picker hierarchy paints status dots and gray short ids."""
    del qapp
    model = ExprPickerModel(_build_controller())
    assert model.rowCount(QModelIndex()) == 1  # file
    file_idx = model.index(0, 0, QModelIndex())
    assert model.data(file_idx, Qt.ItemDataRole.DisplayRole) == "sample.vms"
    assert model.data(file_idx, ComponentColorRole) is not None

    group_idx = model.index(0, 0, file_idx)
    spectrum_idx = model.index(0, 0, group_idx)
    assert model.data(spectrum_idx, ObjectIdPrefixRole) == "s0d8e"
    assert model.data(spectrum_idx, ComponentColorRole) is not None

    region_idx = model.index(0, 0, spectrum_idx)
    assert model.data(region_idx, ObjectIdPrefixRole) == "r3113"

    peak1 = model.index(1, 0, region_idx)  # background is 0, Peak 1 is 1
    assert model.data(peak1, Qt.ItemDataRole.DisplayRole) == "Peak 1"
    assert model.data(peak1, ComponentColorRole) == color_for_component(component_id="pabcd111")
    peak2 = model.index(2, 0, region_idx)
    assert model.data(peak2, ComponentColorRole) == color_for_component(component_id="p12ef222")
    assert model.insert_token_for("pabcd111") == "pabcd"
    assert "pabcd111" in model.component_ids


def test_expr_popup_insert_and_commit(qapp: QApplication) -> None:
    """Clicking a component inserts a short id; commit emits the expression."""
    del qapp
    controller = _build_controller()
    committed: list[str] = []

    popup = ExprEditorPopup(
        controller,
        initial_text="2 * ",
        focus_spectrum_id="s0d8eabc",
        focus_region_id="r3113xyz",
    )
    popup.accepted.connect(committed.append)
    popup.show()
    QApplication.processEvents()

    file_idx = popup._model.index(0, 0, QModelIndex())
    group_idx = popup._model.index(0, 0, file_idx)
    spectrum_idx = popup._model.index(0, 0, group_idx)
    region_idx = popup._model.index(0, 0, spectrum_idx)
    peak_idx = popup._model.index(1, 0, region_idx)
    popup._on_tree_clicked(peak_idx)

    assert popup._expr_edit.text() == "2 * pabcd"
    popup._commit()
    QApplication.processEvents()
    assert committed == ["2 * pabcd"]
    popup.deleteLater()
    QApplication.processEvents()
