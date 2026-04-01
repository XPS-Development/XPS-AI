"""
CSV-like export helpers for ObjectLike projections.
"""

import csv
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np

from core.types import ComponentLike, RegionLike, SpectrumLike
from tools.evaluation import spectrum_bundle


SpectrumRepr = tuple[SpectrumLike, tuple[tuple[RegionLike, tuple[ComponentLike, ...]], ...]]


def export_spectrum_peak_parameters_csv(
    path: str | Path,
    components: tuple[ComponentLike, ...],
    separator: str = ",",
    *,
    use_xps_peak_names: bool = False,
    precision: int | None = None,
) -> None:
    """
    Export peak parameters from a spectrum-level component tuple into CSV.

    Parameters
    ----------
    path : str or Path
        Output file path.
    components : tuple[ComponentLike, ...]
        Components belonging to a spectrum or a selected subset.
    separator : str, optional
        Column separator character.
    use_xps_peak_names : bool, optional
        If True and the peak model is pseudo-voigt, use XPS-style aliases:
        Position, Area, FWHM, and %GL.
    precision : int or None, optional
        Decimal precision for exported numeric values.
    """
    csv_text = _serialize_spectrum_peak_parameters_csv(
        components=components,
        separator=separator,
        use_xps_peak_names=use_xps_peak_names,
        precision=precision,
    )
    _write_text(path, csv_text)


def export_spectrum_csv(
    path: str | Path,
    spectrum_repr: SpectrumRepr,
    separator: str = ",",
    *,
    include_evaluated_components: bool = False,
    include_background: bool = True,
    include_difference: bool = True,
    precision: int | None = None,
) -> None:
    """
    Export spectrum x/y arrays from a spectrum representation as CSV-like text.

    Parameters
    ----------
    spectrum_repr : tuple[SpectrumDTO, tuple[tuple[RegionDTO, tuple[ComponentDTO, ...]], ...]]
        Full spectrum representation. Only the spectrum DTO data are exported.
    separator : str, optional
        Column separator character.
    include_evaluated_components : bool, optional
        If True, add evaluated component columns similar to XPS DAT exports.
    include_background : bool, optional
        If True and ``include_evaluated_components`` is enabled, include the
        evaluated background column.
    include_difference : bool, optional
        If True and ``include_evaluated_components`` is enabled, include the
        difference (raw - model) column.

    """
    csv_text = _serialize_spectrum_csv(
        spectrum_repr=spectrum_repr,
        separator=separator,
        include_evaluated_components=include_evaluated_components,
        include_background=include_background,
        include_difference=include_difference,
        precision=precision,
    )
    _write_text(path, csv_text)


def _serialize_spectrum_csv(
    spectrum_repr: SpectrumRepr,
    separator: str = ",",
    *,
    include_evaluated_components: bool = False,
    include_background: bool = True,
    include_difference: bool = True,
    precision: int | None = None,
) -> str:
    """
    Serialize a spectrum representation as CSV-like text.
    """
    _validate_separator(separator)

    spectrum, region_reprs = spectrum_repr
    stream = StringIO()
    writer = csv.writer(stream, delimiter=separator, lineterminator="\n")
    if not include_evaluated_components:
        writer.writerow(["x", "y"])
        for x_value, y_value in _iter_xy_pairs(spectrum.x, spectrum.y):
            writer.writerow([_format_value(x_value, precision), _format_value(y_value, precision)])
        return stream.getvalue()

    evaluated = spectrum_bundle(spectrum, region_reprs, include_background=include_background)
    max_peak_count = max((len(region.peaks) for region in evaluated.regions), default=0)

    header = ["x", "raw_intensity", "peak_sum"]
    if include_background:
        header.append("background")
    header.extend([f"peak_{index + 1}" for index in range(max_peak_count)])
    if include_difference:
        header.append("difference")
    writer.writerow(header)

    peak_sum_full = np.full(spectrum.x.shape, np.nan, dtype=float)
    background_full = np.full(spectrum.x.shape, np.nan, dtype=float)
    difference_full = np.full(spectrum.x.shape, np.nan, dtype=float)
    peak_full = [np.full(spectrum.x.shape, np.nan, dtype=float) for _ in range(max_peak_count)]

    for region in evaluated.regions:
        index_map = _map_region_points_to_spectrum_indices(spectrum.x, region.x)
        region_peak_sum = np.zeros(region.x.shape, dtype=float)

        for peak_index, peak in enumerate(region.peaks):
            region_peak_sum += peak.y
            peak_full[peak_index][index_map] = peak.y

        peak_sum_full[index_map] = region_peak_sum

        if include_background and region.background is not None:
            background_full[index_map] = region.background.y

        if include_difference:
            difference_full[index_map] = region.residuals

    for idx, (x_value, y_value) in enumerate(_iter_xy_pairs(spectrum.x, spectrum.y)):
        row: list[object] = [
            _format_value(x_value, precision),
            _format_value(y_value, precision),
            _format_value(peak_sum_full[idx], precision),
        ]
        if include_background:
            row.append(_format_value(background_full[idx], precision))
        for peak_values in peak_full:
            row.append(_format_value(peak_values[idx], precision))
        if include_difference:
            row.append(_format_value(difference_full[idx], precision))
        writer.writerow(row)

    return stream.getvalue()


def _serialize_spectrum_peak_parameters_csv(
    components: tuple[ComponentLike, ...],
    separator: str = ",",
    *,
    use_xps_peak_names: bool = False,
    precision: int | None = None,
) -> str:
    """
    Serialize peak parameters from component tuple as CSV-like text.
    """
    _validate_separator(separator)
    peaks = tuple(component for component in components if component.kind == "peak")
    if len(peaks) == 0:
        raise ValueError("components must include at least one peak")

    stream = StringIO()
    writer = csv.writer(stream, delimiter=separator, lineterminator="\n")
    first_names_values = _peak_parameter_values(peaks[0], use_xps_peak_names)
    header_names = [name for name, _ in first_names_values]
    writer.writerow(["peak", *header_names])

    for peak_index, peak in enumerate(peaks, start=1):
        names_values = _peak_parameter_values(peak, use_xps_peak_names)
        names = [name for name, _ in names_values]
        if names != header_names:
            raise ValueError("all exported peaks must have compatible parameter names")
        writer.writerow(
            [f"peak_{peak_index}", *[_format_value(value, precision) for _, value in names_values]]
        )
    return stream.getvalue()


def _iter_xy_pairs(x_values: Iterable[float], y_values: Iterable[float]) -> Iterable[tuple[float, float]]:
    """
    Iterate over x/y pairs and validate equal lengths.
    """
    x_list = list(x_values)
    y_list = list(y_values)
    if len(x_list) != len(y_list):
        raise ValueError("x and y arrays must have equal length")
    return zip(x_list, y_list, strict=True)


def _validate_separator(separator: str) -> None:
    """
    Validate CSV separator.
    """
    if len(separator) != 1:
        raise ValueError("separator must be a single character")


def _map_region_points_to_spectrum_indices(
    spectrum_x: np.ndarray,
    region_x: np.ndarray,
) -> np.ndarray:
    """
    Map region x-values to indices in the parent spectrum x-array.

    Parameters
    ----------
    spectrum_x : NDArray
        Full spectrum x-axis array.
    region_x : NDArray
        Region x-axis array.

    Returns
    -------
    NDArray
        Integer indices into ``spectrum_x`` for each point in ``region_x``.
    """
    index_map: dict[float, list[int]] = {}
    for idx, x_value in enumerate(spectrum_x):
        key = float(x_value)
        index_map.setdefault(key, []).append(idx)

    mapped: list[int] = []
    usage_counter: dict[float, int] = {}
    for x_value in region_x:
        key = float(x_value)
        candidates = index_map.get(key)
        if not candidates:
            raise ValueError("region x-values must be present in spectrum x-values")
        used = usage_counter.get(key, 0)
        if used >= len(candidates):
            raise ValueError("region x-values mapping is ambiguous for the parent spectrum")
        mapped.append(candidates[used])
        usage_counter[key] = used + 1

    return np.asarray(mapped, dtype=int)


def _pseudo_voigt_xps_alias_values(component: ComponentLike) -> list[tuple[str, float]]:
    """
    Return pseudo-voigt parameters as XPS-style aliases.

    Parameters
    ----------
    component : ComponentLike
        Peak component with pseudo-voigt parameters.

    Returns
    -------
    list[tuple[str, float]]
        Aliased parameter names and values.
    """
    amp = component.parameters["amp"].value
    cen = component.parameters["cen"].value
    sig = component.parameters["sig"].value
    frac = component.parameters["frac"].value
    return [
        ("Position", cen),
        ("Area", amp),
        ("FWHM", 2.0 * sig),
        ("%GL", 100.0 * frac),
    ]


def _peak_parameter_values(component: ComponentLike, use_xps_peak_names: bool) -> list[tuple[str, float]]:
    """
    Return export parameter/value pairs for a peak component.
    """
    if component.kind != "peak":
        raise ValueError("components must contain only peak entries for peak export")
    if use_xps_peak_names and component.model.name == "pseudo-voigt":
        return _pseudo_voigt_xps_alias_values(component)
    return [(name, parameter.value) for name, parameter in component.parameters.items()]


def _write_text(path: str | Path, text: str) -> None:
    """
    Write UTF-8 text to disk, creating parent directories as needed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _format_value(value: object, precision: int | None) -> object:
    """
    Format numeric value with optional precision.
    """
    if precision is None:
        return value
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return round(float(value), precision)
    return value
    return value
