"""Tests for application settings dropdowns."""

from __future__ import annotations

import sys
from typing import cast

import pytest
from PySide6.QtWidgets import QApplication

from app.parameters import AppParameters
from core.math_models import ModelRegistry
from ui.options_dialog import OptionsDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return cast(QApplication, app)


def test_model_and_serialization_mode_are_dropdowns(qapp: QApplication) -> None:
    """Peak, background, and save mode are chosen from lists, not free text."""
    del qapp
    params = AppParameters(
        default_peak_model="tail-pseudo-voigt",
        default_background_model="linear",
        default_serialization_mode="append",
    )
    dialog = OptionsDialog(
        peak_model_names=ModelRegistry.get_peak_model_names(),
        background_model_names=ModelRegistry.get_background_model_names(),
    )
    dialog.load_from_params(params)

    assert dialog._default_peak_model_combo.currentData() == "tail-pseudo-voigt"
    assert dialog._default_bg_model_combo.currentData() == "linear"
    assert dialog._serialization_mode_combo.currentData() == "append"
    assert "asym-pseudo-voigt" in [
        dialog._default_peak_model_combo.itemData(i)
        for i in range(dialog._default_peak_model_combo.count())
    ]

    dialog._default_peak_model_combo.setCurrentIndex(
        dialog._default_peak_model_combo.findData("pseudo-voigt")
    )
    dialog._serialization_mode_combo.setCurrentIndex(
        dialog._serialization_mode_combo.findData("replace")
    )
    dialog.apply_to_params(params)
    assert params.default_peak_model == "pseudo-voigt"
    assert params.default_serialization_mode == "replace"
    assert params.invert_x_axis is True

    dialog._invert_x_axis_cb.setChecked(False)
    dialog.apply_to_params(params)
    assert params.invert_x_axis is False
