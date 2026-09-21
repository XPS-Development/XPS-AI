"""Tests for :class:`app.usecases.export.ExportUseCases`."""

from app.csv_export import CSVExportService
from app.query_service import QueryService
from app.usecases.export import ExportUseCases
from core.services import CoreContext


def test_export_peak_parameters_writes_csv(simple_collection, spectrum_id: str, tmp_path) -> None:
    """export_peak_parameters writes a non-empty peak-parameter table."""
    ctx = CoreContext.from_collection(simple_collection)
    export = ExportUseCases(QueryService(ctx), CSVExportService())
    output = tmp_path / "peaks.csv"

    export.export_peak_parameters(spectrum_id, output)

    text = output.read_text(encoding="utf-8")
    assert "amp" in text
    assert len(text.strip().splitlines()) >= 2


def test_export_spectrum_writes_xy_csv(simple_collection, spectrum_id: str, tmp_path) -> None:
    """export_spectrum writes spectrum x/y columns."""
    ctx = CoreContext.from_collection(simple_collection)
    export = ExportUseCases(QueryService(ctx), CSVExportService())
    output = tmp_path / "spectrum.csv"

    export.export_spectrum(spectrum_id, output)

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    assert "x" in lines[0].lower() or "," in lines[0]
