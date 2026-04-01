from pathlib import Path

from app.csv_export import CSVExportService
from app.orchestration import AppOrchestrator, AppParameters


def test_csv_export_service_exports_peak_file(dto_service, peak_id: str, tmp_path: Path) -> None:
    service = CSVExportService()
    peak_dto = dto_service.get_component(peak_id)
    output = tmp_path / "peak.csv"

    service.export_spectrum_peak_parameters(output, (peak_dto,))

    text = output.read_text(encoding="utf-8")
    assert text.startswith("peak,")


def test_csv_export_service_exports_spectrum_file(
    dto_service, spectrum_id: str, tmp_path: Path
) -> None:
    service = CSVExportService()
    spectrum_repr = dto_service.get_spectrum_repr(spectrum_id)
    output = tmp_path / "spectrum.csv"

    service.export_spectrum(output, spectrum_repr)

    text = output.read_text(encoding="utf-8")
    assert text.startswith("x,y")


def test_orchestrator_exports_peak_parameters(simple_collection, peak_id: str, tmp_path: Path) -> None:
    orchestrator = AppOrchestrator(simple_collection, AppParameters())
    output = tmp_path / "orchestrator_peak.csv"

    orchestrator.export_peak_parameters("s1", output, normalized=False)

    assert output.exists()
    assert output.read_text(encoding="utf-8").splitlines()[0].startswith("peak,")


def test_orchestrator_exports_peak_parameters_with_xps_aliases(
    simple_collection, peak_id: str, tmp_path: Path
) -> None:
    orchestrator = AppOrchestrator(simple_collection, AppParameters())
    output = tmp_path / "orchestrator_peak_xps.csv"
    orchestrator.export_peak_parameters(
        "s1",
        output,
        normalized=False,
        use_xps_peak_names=True,
    )
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "peak,Position,Area,FWHM,%GL"
    assert lines[1].startswith("peak_1,")


def test_orchestrator_exports_spectrum(simple_collection, spectrum_id: str, tmp_path: Path) -> None:
    orchestrator = AppOrchestrator(simple_collection, AppParameters())
    output = tmp_path / "orchestrator_spectrum.csv"

    orchestrator.export_spectrum(spectrum_id, output, normalized=True, separator=";")

    assert output.exists()
    assert output.read_text(encoding="utf-8").splitlines()[0] == "x;y"
