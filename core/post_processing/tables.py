"""Pure detailed and analytical projections of saved matching results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from core.dto import AnalysisCell, AnalysisColumn, AnalysisRow, AnalysisTable
from core.post_processing.metrics import MetricDefinition, calculate_metrics

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.post_processing.state import PostProcessingSeries

TableView = Literal["components"]


def build_table(
    series: PostProcessingSeries, names: Mapping[str, str], view: TableView = "components"
) -> AnalysisTable:
    """Build complete rows in snapshot order, retaining empty and unavailable steps."""
    if view not in ("components",):
        raise ValueError(f"Unknown post-processing view: {view}")
    base = (AnalysisColumn("step", "Step"), AnalysisColumn("spectrum", "Spectrum"))
    result = series.result
    if result is None:
        return AnalysisTable(base, ())
    track_names = dict(series.track_names)
    columns = (
        *base,
        *(
            AnalysisColumn(id_, label, unit)
            for id_, label, unit in (
                ("region", "Region ID", ""),
                ("component", "Component ID", ""),
                ("track", "Track ID", ""),
                ("track_name", "Track name", ""),
                ("model", "Model", ""),
                ("position", "Position", "eV"),
                ("area", "Area", "area"),
                ("fwhm", "FWHM", "eV"),
                ("sig", "sig", "eV"),
                ("fraction", "Area / sum of spectrum peak areas", "%"),
                ("link", "Link status", ""),
            )
        ),
    )
    by_component = {cid: track for track in result.tracks for cid in track.component_ids}
    fractions = {
        (v.spectrum_id, v.metric_id): v
        for v in calculate_metrics(
            result,
            tuple(
                MetricDefinition(
                    track.id_, "Spectrum area fraction", "spectrum_area_fraction", (track.id_,)
                )
                for track in result.tracks
            ),
        )
    }
    rows = []
    for index, step in enumerate(result.series.steps, 1):
        identity = (
            AnalysisCell(index),
            AnalysisCell(names.get(step.spectrum_id, step.spectrum_id)),
        )
        for peak in step.peaks:
            track = by_component[peak.component_id]
            incoming = next(
                (link for link in track.links if link.target_id == peak.component_id), None
            )
            status = (
                "start"
                if incoming is None
                else "manual"
                if incoming.manual
                else "ambiguous"
                if incoming.ambiguous
                else "automatic"
            )
            fraction = fractions[step.spectrum_id, track.id_]
            numerical = tuple(
                AnalysisCell(value, "ambiguous" if track.ambiguous else "ok")
                if value is not None
                else AnalysisCell(None, "missing", "missing_feature")
                for value in (peak.position, peak.area, peak.fwhm, peak.sig)
            )
            cells = (
                *identity,
                AnalysisCell(peak.region_id),
                AnalysisCell(peak.component_id),
                AnalysisCell(track.id_),
                AnalysisCell(track_names.get(track.anchor_id, "")),
                AnalysisCell(peak.model_name),
                *numerical,
                AnalysisCell(fraction.value, fraction.status, fraction.reason),
                AnalysisCell(status),
            )
            rows.append(
                AnalysisRow(index, step.spectrum_id, cells, peak.region_id, peak.component_id)
            )
        for excluded in step.excluded:
            if excluded.reason == "background":
                continue
            values = (
                *identity,
                AnalysisCell(excluded.region_id),
                AnalysisCell(excluded.component_id),
                *(AnalysisCell(None, "missing", excluded.reason) for _ in range(len(columns) - 5)),
                AnalysisCell("excluded", "missing", excluded.reason),
            )
            rows.append(
                AnalysisRow(
                    index, step.spectrum_id, values, excluded.region_id, excluded.component_id
                )
            )
        if not step.peaks and not any(e.reason != "background" for e in step.excluded):
            reason = "empty_step" if step.available else "unavailable_spectrum"
            cells = (*identity, *(AnalysisCell(None, "missing", reason) for _ in columns[2:]))
            rows.append(AnalysisRow(index, step.spectrum_id, cells))
    return AnalysisTable(columns, tuple(rows))


def build_trend_tables(
    series: PostProcessingSeries, names: Mapping[str, str], track_id: str
) -> tuple[AnalysisTable, AnalysisTable]:
    """Project a selected trend at every supplied step, preserving missing positions."""
    detailed = build_table(series, names)
    if series.result is None:
        return detailed, AnalysisTable((), ())
    track = next(t for t in series.result.tracks if t.id_ == track_id)
    by_step = {r.spectrum_id: r for r in detailed.rows if r.component_id in track.component_ids}
    rows = []
    for index, step in enumerate(series.result.series.steps, 1):
        row = by_step.get(step.spectrum_id)
        if row is None:
            row = AnalysisRow(
                index,
                step.spectrum_id,
                (
                    AnalysisCell(index),
                    AnalysisCell(names.get(step.spectrum_id, step.spectrum_id)),
                    *(AnalysisCell(None, "missing", "missing_peak") for _ in detailed.columns[2:]),
                ),
            )
        rows.append(row)
    indices = [
        i
        for i, column in enumerate(detailed.columns)
        if column.id_ in ("step", "spectrum", "position", "area", "fwhm", "sig", "fraction")
    ]
    preview_columns = tuple(
        AnalysisColumn(c.id_, c.label, c.unit, track_id, track.anchor_id) if i > 1 else c
        for i, c in enumerate(detailed.columns)
        if i in indices
    )
    preview_rows = tuple(
        AnalysisRow(
            r.step, r.spectrum_id, tuple(r.cells[i] for i in indices), r.region_id, r.component_id
        )
        for r in rows
    )
    return AnalysisTable(detailed.columns, tuple(rows)), AnalysisTable(
        preview_columns, preview_rows
    )
