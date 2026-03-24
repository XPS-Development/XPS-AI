from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication, QMenu, QWidget

from ui.context_menus import attach_region_context_actions, attach_spectrum_context_actions


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_spectrum_context_menu_exports_csv(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    del qapp
    controller = MagicMock()
    controller.selected_spectrum_id = "s1"
    controller.query.get_regions_ids.return_value = ()

    menu = QMenu()
    actions = attach_spectrum_context_actions(menu, controller)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "ui.context_menus.run_export_spectrum_pipeline",
        lambda _controller, spectrum_ids, parent=None: calls.append(spectrum_ids) or True,
    )

    actions._on_export_spectrum_csv()

    assert calls == [["s1"]]


def test_region_context_menu_exports_first_peak(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    del qapp
    controller = MagicMock()
    controller.query.get_background_id.return_value = None
    controller.selected_spectrum_id = "s1"

    menu = QMenu()
    actions = attach_region_context_actions(menu, controller, "r1", QWidget())
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "ui.context_menus.run_export_peaks_pipeline",
        lambda _controller, spectrum_ids, parent=None: calls.append(spectrum_ids) or True,
    )

    actions._on_export_peak_csv()

    assert calls == [["s1"]]
