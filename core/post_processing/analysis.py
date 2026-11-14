"""Independent analytical tables with explicit cross-series point alignment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import TYPE_CHECKING

from core.dto import AnalysisCell, AnalysisColumn, AnalysisRow, AnalysisTable, SourceLocation
from core.post_processing.labels import trend_label
from core.post_processing.metrics import MetricDefinition
from core.post_processing.state import new_revision

if TYPE_CHECKING:
    from collections.abc import Mapping

    from core.post_processing.metrics import MetricOperation
    from core.post_processing.state import PostProcessingSeries


@dataclass(frozen=True)
class TrendSource:
    """One chosen trend; identity is qualified by its owning series."""

    id_: str
    series_id: str
    track_id: str
    anchor_id: str

    def __post_init__(self) -> None:
        """Validate qualified identities without resolving a possibly missing series."""
        if (
            any(
                not isinstance(value, str) or not value
                for value in (self.id_, self.series_id, self.track_id, self.anchor_id)
            )
            or self.track_id != f"track/{self.anchor_id}"
        ):
            raise ValueError("Invalid trend source identity")


@dataclass(frozen=True)
class PointReference:
    """Saved source point; a missing peak keeps its spectrum position."""

    source_id: str
    spectrum_id: str | None
    component_id: str | None

    def __post_init__(self) -> None:
        """Reject malformed identities while preserving explicit absent points."""
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("A point requires a source ID")
        for value in (self.spectrum_id, self.component_id):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("Point IDs must be nonempty strings or absent")
        if self.component_id is not None and self.spectrum_id is None:
            raise ValueError("A component reference requires a spectrum")


@dataclass(frozen=True)
class AlignedRow:
    """Stable row identity and explicit references, independent of view sorting."""

    id_: str
    points: tuple[PointReference, ...]


@dataclass(frozen=True)
class TableMetric:
    """Column operation on source IDs with an explicitly identified reference row."""

    id_: str
    name: str
    operation: MetricOperation
    tracks: tuple[str, ...]
    denominator: tuple[str, ...] = ()
    reference_row_id: str | None = None

    def __post_init__(self) -> None:
        """Reuse the fixed operation grammar, without accepting formula chains."""
        if any(not isinstance(value, str) or not value.strip() for value in (self.id_, self.name)):
            raise ValueError("Column ID and name must be nonempty strings")
        self.definition()

    def definition(self) -> MetricDefinition:
        """Return validated arithmetic metadata with the row ID as the reference key."""
        return MetricDefinition(
            self.id_,
            self.name,
            self.operation,
            self.tracks,
            self.denominator,
            self.reference_row_id,
        )

    @property
    def unit(self) -> str:
        """Return the column's physical unit."""
        return self.definition().unit


@dataclass(frozen=True)
class AnalyticalTableState:
    """Saved analysis, owned by the document rather than by a spectrum series."""

    id_: str
    name: str
    sources: tuple[TrendSource, ...] = ()
    rows: tuple[AlignedRow, ...] = ()
    metrics: tuple[TableMetric, ...] = ()
    column_order: tuple[str, ...] = ()
    plotted_columns: tuple[str, ...] = ()
    dependencies: tuple[tuple[str, str], ...] = ()
    revision: str = field(default_factory=new_revision)

    def __post_init__(self) -> None:
        """Validate structural identities while retaining unresolved external references."""
        if any(
            not isinstance(v, str) or not v.strip() for v in (self.id_, self.name, self.revision)
        ):
            raise ValueError("Table ID, name and revision must be nonempty")
        for ids in (
            [s.id_ for s in self.sources],
            [r.id_ for r in self.rows],
            [m.id_ for m in self.metrics],
            list(self.column_order),
            list(self.plotted_columns),
            [key for key, _ in self.dependencies],
        ):
            if any(not isinstance(v, str) or not v for v in ids) or len(ids) != len(set(ids)):
                raise ValueError("Analytical table identities must be nonempty and unique")
        if len({(s.series_id, s.track_id) for s in self.sources}) != len(self.sources):
            raise ValueError("A trend can only be added once to the same table")
        for source in self.sources:
            if (
                not source.series_id
                or not source.anchor_id
                or source.track_id != f"track/{source.anchor_id}"
            ):
                raise ValueError("Invalid trend source identity")
        source_ids = {s.id_ for s in self.sources}
        for row in self.rows:
            ids = [p.source_id for p in row.points]
            if set(ids) != source_ids or len(ids) != len(set(ids)):
                raise ValueError("Each row must contain exactly one point per source")
            if any(p.component_id is not None and p.spectrum_id is None for p in row.points):
                raise ValueError("A component reference requires a spectrum")


def dependency_stamp(series: PostProcessingSeries) -> str:
    """Hash scientific series state, excluding names and presentation edits."""
    payload = (
        series.spectrum_ids,
        asdict(series.settings),
        asdict(series.constraints),
        series.input_fingerprint,
        None if series.result is None else asdict(series.result),
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def align_rows(
    sources: tuple[TrendSource, ...],
    series: Mapping[str, PostProcessingSeries],
    previous: tuple[AlignedRow, ...] = (),
) -> tuple[AlignedRow, ...]:
    """Pair supplied steps by order to the longest series, never compressing gaps."""
    length = max(
        (len(series[s.series_id].spectrum_ids) for s in sources if s.series_id in series), default=0
    )
    rows = []
    used = set()
    for index in range(length):
        points = []
        for source in sources:
            saved = series.get(source.series_id)
            sid = (
                None
                if saved is None or index >= len(saved.spectrum_ids)
                else saved.spectrum_ids[index]
            )
            cid = None
            if saved is not None and saved.result is not None:
                track = next(
                    (
                        t
                        for t in saved.result.tracks
                        if t.id_ == source.track_id and t.anchor_id == source.anchor_id
                    ),
                    None,
                )
                step = next((s for s in saved.result.series.steps if s.spectrum_id == sid), None)
                if track is not None and step is not None:
                    cid = next(
                        (
                            p.component_id
                            for p in step.peaks
                            if p.component_id in track.component_ids
                        ),
                        None,
                    )
            points.append(PointReference(source.id_, sid, cid))
        current = {p.source_id: p.spectrum_id for p in points}
        row_id = None
        for old in previous:
            shared = [p for p in old.points if p.source_id in current]
            if (
                old.id_ not in used
                and shared
                and all(current[p.source_id] == p.spectrum_id for p in shared)
            ):
                row_id = old.id_
                break
        row_id = row_id or new_revision()
        used.add(row_id)
        rows.append(AlignedRow(row_id, tuple(points)))
    return tuple(rows)


class _Missing(Exception):
    pass


def build_analysis(
    table: AnalyticalTableState,
    series: Mapping[str, PostProcessingSeries],
    names: Mapping[str, str],
    available: set[str] | None = None,
) -> AnalysisTable:
    """Calculate one full precision projection shared by UI, plots and CSV."""
    source_map = {s.id_: s for s in table.sources}
    rows = {r.id_: {p.source_id: p for p in r.points} for r in table.rows}
    metric_map = {m.id_: m for m in table.metrics}
    metrics = [
        metric_map[mid]
        for mid in dict.fromkeys((*table.column_order, *metric_map))
        if mid in metric_map
    ]
    base = [AnalysisColumn("step", "Row"), AnalysisColumn("row_id", "Row ID")]
    for source in table.sources:
        saved = series.get(source.series_id)
        label = (
            "Unavailable trend"
            if saved is None
            else f"{saved.name} / {trend_label(saved, source.track_id)}"
        )
        base.append(
            AnalysisColumn(
                f"source/{source.id_}",
                label + " — spectrum",
                track_id=source.id_,
                anchor_id=source.anchor_id,
            )
        )
    columns = (
        *base,
        *(
            AnalysisColumn(
                m.id_,
                m.name,
                m.unit,
                m.tracks[0] if len(m.tracks) == 1 else None,
                source_map[m.tracks[0]].anchor_id
                if len(m.tracks) == 1 and m.tracks[0] in source_map
                else None,
                m.definition().quantity,
            )
            for m in metrics
        ),
    )
    result_rows = []
    for index, aligned in enumerate(table.rows, 1):
        locations: list[SourceLocation] = []
        ambiguous = False

        def read(
            source_id: str, row_id: str, field: str, collected: list[SourceLocation] = locations
        ) -> float:
            nonlocal ambiguous
            source = source_map.get(source_id)
            if source is None:
                raise _Missing("operand_selection_required")
            saved = series.get(source.series_id)
            if saved is None or saved.result is None:
                raise _Missing("unavailable_series")
            track = next(
                (
                    t
                    for t in saved.result.tracks
                    if t.id_ == source.track_id and t.anchor_id == source.anchor_id
                ),
                None,
            )
            if track is None:
                raise _Missing("operand_selection_required")
            if row_id not in rows:
                raise _Missing("reference_selection_required")
            point = rows[row_id].get(source_id)
            if point is None or point.spectrum_id is None:
                raise _Missing("missing_step")
            if available is not None and point.spectrum_id not in available:
                raise _Missing("unavailable_spectrum")
            step = next(
                (s for s in saved.result.series.steps if s.spectrum_id == point.spectrum_id), None
            )
            if step is None or not step.available:
                raise _Missing("unavailable_spectrum")
            peak = next(
                (
                    p
                    for p in step.peaks
                    if p.component_id == point.component_id
                    and p.component_id in track.component_ids
                ),
                None,
            )
            if peak is None:
                raise _Missing("missing_peak" if point.component_id is None else "point_changed")
            ambiguous |= track.ambiguous
            location = SourceLocation(
                source.series_id,
                source.track_id,
                step.spectrum_id,
                peak.region_id,
                peak.component_id,
            )
            if location not in collected:
                collected.append(location)
            value = getattr(peak, field) if field != "spectrum_area_fraction" else None
            if field == "spectrum_area_fraction":
                total = number(sum(number(p.area) for p in step.peaks))
                value = 100 * ratio(number(peak.area), total)
            return number(value)

        def number(value: float | None) -> float:
            if value is None:
                raise _Missing("missing_feature")
            if not isfinite(value):
                raise _Missing("nonfinite_value")
            return value

        def ratio(a: float, b: float) -> float:
            if b == 0:
                raise _Missing("zero_denominator")
            return number(a / b)

        def area_sum(ids: tuple[str, ...], row_id: str = aligned.id_) -> float:
            return number(sum(read(s, row_id, "area") for s in dict.fromkeys(ids)))

        cells = [AnalysisCell(index), AnalysisCell(aligned.id_)]
        for point in aligned.points:
            source = source_map[point.source_id]
            saved = series.get(source.series_id)
            peak = (
                None
                if saved is None or saved.result is None
                else next(
                    (
                        peak
                        for step in saved.result.series.steps
                        for peak in step.peaks
                        if peak.component_id == point.component_id
                    ),
                    None,
                )
            )
            provenance = (
                ()
                if peak is None or point.spectrum_id is None
                else (
                    SourceLocation(
                        source.series_id,
                        source.track_id,
                        point.spectrum_id,
                        peak.region_id,
                        peak.component_id,
                    ),
                )
            )
            cells.append(
                AnalysisCell(
                    names.get(point.spectrum_id, point.spectrum_id) if point.spectrum_id else None,
                    "ok" if point.component_id else "missing",
                    None if point.component_id else "missing_peak",
                    provenance,
                )
            )
        for metric in metrics:
            locations.clear()
            ambiguous = False
            try:
                op, ids = metric.operation, metric.tracks
                if op == "area_sum":
                    value = area_sum(ids)
                elif op == "area_sum_ratio":
                    value = ratio(area_sum(ids), area_sum(metric.denominator))
                elif op == "area_ratio":
                    value = ratio(
                        read(ids[0], aligned.id_, "area"), read(ids[1], aligned.id_, "area")
                    )
                elif op == "area_fraction":
                    value = 100 * ratio(
                        read(ids[0], aligned.id_, "area"), area_sum(metric.denominator)
                    )
                elif op == "position_difference":
                    value = read(ids[0], aligned.id_, "position") - read(
                        ids[1], aligned.id_, "position"
                    )
                elif op == "position_shift":
                    value = read(ids[0], aligned.id_, "position") - read(
                        ids[0], metric.reference_row_id or "", "position"
                    )
                else:
                    value = read(ids[0], aligned.id_, op)
                cells.append(
                    AnalysisCell(
                        number(value), "ambiguous" if ambiguous else "ok", sources=tuple(locations)
                    )
                )
            except _Missing as error:
                cells.append(AnalysisCell(None, "missing", str(error), tuple(locations)))
        first = next((p.spectrum_id for p in aligned.points if p.spectrum_id), "")
        result_rows.append(AnalysisRow(index, first, tuple(cells), id_=aligned.id_))
    return AnalysisTable(tuple(columns), tuple(result_rows))
