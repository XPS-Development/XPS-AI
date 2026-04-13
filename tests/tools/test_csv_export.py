import numpy as np
import pytest

from tools.csv_export import export_spectrum_csv, export_spectrum_peak_parameters_csv


def test_export_peak_parameters_csv_returns_header_and_rows(
    dto_service, peak_id: str, tmp_path
) -> None:
    peak_dto = dto_service.get_component(peak_id, normalized=False)
    output = tmp_path / "peak_params.csv"

    export_spectrum_peak_parameters_csv(output, (peak_dto,))
    csv_text = output.read_text(encoding="utf-8")
    lines = csv_text.strip().splitlines()

    assert len(lines) == 2
    assert lines[0].startswith("peak,")
    assert "cen" in lines[0]
    assert lines[1].startswith("peak_1,")


def test_export_peak_parameters_csv_respects_separator(dto_service, peak_id: str, tmp_path) -> None:
    peak_dto = dto_service.get_component(peak_id, normalized=True)
    output = tmp_path / "peak_params_sep.csv"

    export_spectrum_peak_parameters_csv(output, (peak_dto,), separator=";")
    csv_text = output.read_text(encoding="utf-8")

    assert csv_text.splitlines()[0].startswith("peak;")
    assert "," not in csv_text.splitlines()[0]


def test_export_peak_parameters_csv_raises_for_non_peak(dto_service, background_id: str, tmp_path) -> None:
    background_dto = dto_service.get_component(background_id)
    output = tmp_path / "non_peak.csv"

    with pytest.raises(ValueError, match="at least one peak"):
        export_spectrum_peak_parameters_csv(output, (background_dto,))


def test_export_peak_parameters_csv_uses_xps_aliases_for_pseudo_voigt(
    dto_service, peak_id: str, tmp_path
) -> None:
    peak_dto = dto_service.get_component(peak_id, normalized=False)
    output = tmp_path / "peak_params_xps.csv"
    export_spectrum_peak_parameters_csv(output, (peak_dto,), use_xps_peak_names=True)
    csv_text = output.read_text(encoding="utf-8")
    header, row = csv_text.strip().splitlines()

    assert header == "peak,Position,Area,FWHM,%GL"
    values = row.split(",")
    assert values[0] == "peak_1"
    assert pytest.approx(float(values[1])) == peak_dto.parameters["cen"].value
    assert pytest.approx(float(values[2])) == peak_dto.parameters["amp"].value
    assert pytest.approx(float(values[3])) == 2.0 * peak_dto.parameters["sig"].value
    assert pytest.approx(float(values[4])) == 100.0 * peak_dto.parameters["frac"].value


def test_export_spectrum_csv_returns_header_and_xy_rows(dto_service, spectrum_id: str, tmp_path) -> None:
    spectrum_repr = dto_service.get_spectrum_repr(spectrum_id, normalized=False)
    spectrum_dto = spectrum_repr[0]

    output = tmp_path / "spectrum.csv"
    export_spectrum_csv(output, spectrum_repr)
    csv_text = output.read_text(encoding="utf-8")
    lines = csv_text.strip().splitlines()

    assert lines[0] == "x,y"
    assert len(lines) == spectrum_dto.x.size + 1


def test_export_spectrum_csv_respects_separator(dto_service, spectrum_id: str, tmp_path) -> None:
    spectrum_repr = dto_service.get_spectrum_repr(spectrum_id, normalized=True)

    output = tmp_path / "spectrum_tab.csv"
    export_spectrum_csv(output, spectrum_repr, separator="\t")
    csv_text = output.read_text(encoding="utf-8")

    lines = csv_text.strip().splitlines()
    assert lines
    assert lines[0] == "x\ty"


def test_export_spectrum_csv_raises_for_size_mismatch(dto_service, spectrum_id: str, tmp_path) -> None:
    spectrum_repr = dto_service.get_spectrum_repr(spectrum_id, normalized=False)
    spectrum_dto, reg_repr = spectrum_repr
    broken_spectrum = type(spectrum_dto)(
        id_=spectrum_dto.id_,
        parent_id=spectrum_dto.parent_id,
        normalized=spectrum_dto.normalized,
        x=spectrum_dto.x,
        y=np.array([1.0, 2.0]),
    )

    with pytest.raises(ValueError, match="equal length"):
        export_spectrum_csv(tmp_path / "broken.csv", (broken_spectrum, reg_repr))


def test_export_spectrum_csv_includes_evaluated_columns(dto_service, spectrum_id: str, tmp_path) -> None:
    spectrum_repr = dto_service.get_spectrum_repr(spectrum_id, normalized=False)

    output = tmp_path / "spectrum_eval.csv"
    export_spectrum_csv(
        output,
        spectrum_repr,
        include_evaluated_components=True,
    )
    csv_text = output.read_text(encoding="utf-8")
    lines = csv_text.strip().splitlines()

    assert lines
    header = lines[0].split(",")
    assert "raw_intensity" in header
    assert "peak_sum" in header
    assert "background" in header
    assert "peak_1" in header
    assert "difference" in header
    assert "x" in header


def test_export_spectrum_csv_can_disable_background_and_difference(
    dto_service, spectrum_id: str, tmp_path
) -> None:
    spectrum_repr = dto_service.get_spectrum_repr(spectrum_id, normalized=False)

    output = tmp_path / "spectrum_eval_min.csv"
    export_spectrum_csv(
        output,
        spectrum_repr,
        include_evaluated_components=True,
        include_background=False,
        include_difference=False,
        separator=";",
    )
    csv_text = output.read_text(encoding="utf-8")
    header = csv_text.strip().splitlines()[0].split(";")

    assert "background" not in header
    assert "difference" not in header
    assert "peak_sum" in header
    assert "spectrum_id" not in header
