"""
Tests for :class:`ui.controller.ControllerWrapper` granular Qt signals.
"""

import sys
from collections import defaultdict

import pytest
from PySide6.QtWidgets import QApplication

from ui.controller import ControllerWrapper


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for QObject signal tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _connect_signal_counts(ctrl: ControllerWrapper) -> defaultdict[str, int]:
    """Connect controller signals to a counter dict and return it."""
    counts: defaultdict[str, int] = defaultdict(int)

    def bump(key: str) -> None:
        counts[key] += 1

    ctrl.spectrumHierarchyChanged.connect(lambda: bump("hierarchy"))
    ctrl.plotNeedsRefresh.connect(lambda: bump("plot"))
    ctrl.propertiesNeedsRefresh.connect(lambda: bump("properties"))
    ctrl.documentStateChanged.connect(lambda: bump("document"))
    ctrl.undoRedoStateChanged.connect(lambda *_a: bump("undo_redo"))
    return counts


def test_update_parameter_emits_plot_properties_not_hierarchy(
    qapp: QApplication,
    simple_collection,
    peak_id: str,
) -> None:
    """Parameter edits should refresh plot and properties, not the spectrum tree."""
    del qapp
    ctrl = ControllerWrapper(collection=simple_collection)
    counts = _connect_signal_counts(ctrl)

    ctrl.update_parameter(peak_id, "amp", "value", 2.0, normalized=False)

    assert counts["hierarchy"] == 0
    assert counts["plot"] == 1
    assert counts["properties"] == 1
    assert counts["document"] == 1
    assert counts["undo_redo"] == 1


def test_emit_full_ui_refresh_emits_all(
    qapp: QApplication,
    simple_collection,
) -> None:
    """Full refresh should notify tree, plot, properties, and document."""
    del qapp
    ctrl = ControllerWrapper(collection=simple_collection)
    counts = _connect_signal_counts(ctrl)

    ctrl.emit_full_ui_refresh()

    assert counts["hierarchy"] == 1
    assert counts["plot"] == 1
    assert counts["properties"] == 1
    assert counts["document"] == 1
    assert counts["undo_redo"] == 1


def test_undo_parameter_does_not_emit_hierarchy(
    qapp: QApplication,
    simple_collection,
    peak_id: str,
) -> None:
    """Undoing a parameter change should not refresh the spectrum hierarchy."""
    del qapp
    ctrl = ControllerWrapper(collection=simple_collection)
    counts = _connect_signal_counts(ctrl)

    ctrl.update_parameter(peak_id, "amp", "value", 2.0, normalized=False)
    counts.clear()
    ctrl.undo()

    assert counts["hierarchy"] == 0
    assert counts["plot"] == 1
    assert counts["properties"] == 1
    assert counts["document"] == 1
    assert counts["undo_redo"] == 1


def test_controller_export_spectrum_forwards_to_orchestrator(
    qapp: QApplication,
    simple_collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del qapp
    ctrl = ControllerWrapper(collection=simple_collection)
    calls: list[tuple[str, str]] = []

    def fake_export_spectrum(spectrum_id: str, path: str, **_kwargs) -> None:
        calls.append((spectrum_id, path))

    monkeypatch.setattr(ctrl.orchestrator, "export_spectrum", fake_export_spectrum)
    ctrl.export_spectrum("s1", "/tmp/out.csv")
    assert calls == [("s1", "/tmp/out.csv")]


def test_controller_export_peak_forwards_to_orchestrator(
    qapp: QApplication,
    simple_collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del qapp
    ctrl = ControllerWrapper(collection=simple_collection)
    calls: list[tuple[str, str]] = []

    def fake_export_peak(spectrum_id: str, path: str, **_kwargs) -> None:
        calls.append((spectrum_id, path))

    monkeypatch.setattr(ctrl.orchestrator, "export_peak_parameters", fake_export_peak)
    ctrl.export_peak_parameters("s1", "/tmp/p.csv")
    assert calls == [("s1", "/tmp/p.csv")]
