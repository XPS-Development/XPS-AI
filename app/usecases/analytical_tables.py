"""Undoable source selection, row alignment and formulas for independent tables."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

from app.command.changes import SetAnalyticalTable
from core.post_processing.analysis import (
    AnalyticalTableState,
    TableMetric,
    TrendSource,
    align_rows,
    dependency_stamp,
)
from core.post_processing.state import new_revision

if TYPE_CHECKING:
    from app.query_service import QueryService
    from core.post_processing.metrics import MetricOperation


class AnalyticalTableUseCases:
    """Produce one command for each saved table edit, including alignment refresh."""

    def __init__(self, query: QueryService) -> None:
        self.query = query

    def create(self, name: str, selections: tuple[tuple[str, str], ...]) -> SetAnalyticalTable:
        """Create a table from independently selected, qualified trends."""
        sources = self._sources(selections)
        table = AnalyticalTableState(new_revision(), name.strip(), sources)
        table = self._align(table)
        columns = tuple(
            TableMetric(new_revision(), self._label(s) + " — E", "position", (s.id_,))
            for s in sources
        )
        table = replace(
            table,
            metrics=columns,
            column_order=tuple(m.id_ for m in columns),
            plotted_columns=tuple(m.id_ for m in columns),
        )
        return SetAnalyticalTable(table.id_, table)

    def _label(self, source: TrendSource) -> str:
        series = self.query.get_post_processing_series(source.series_id).series
        return f"{series.name} / {dict(series.track_names).get(source.anchor_id, source.track_id)}"

    def _sources(self, selections: tuple[tuple[str, str], ...]) -> tuple[TrendSource, ...]:
        sources = []
        for series_id, track_id in dict.fromkeys(selections):
            dto = self.query.get_post_processing_series(series_id)
            if not dto.exportable or dto.series.result is None:
                raise ValueError("Match the series and resolve conflicts before selecting trends")
            track = next((t for t in dto.series.result.tracks if t.id_ == track_id), None)
            if track is None:
                raise ValueError("Unknown trend")
            sources.append(TrendSource(new_revision(), series_id, track_id, track.anchor_id))
        return tuple(sources)

    def _align(self, table: AnalyticalTableState) -> AnalyticalTableState:
        saved = {
            sid: self.query.get_post_processing_series(sid).series
            for sid in self.query.get_post_processing_series_ids()
        }
        for source in table.sources:
            if (
                source.series_id not in saved
                or not self.query.get_post_processing_series(source.series_id).exportable
            ):
                raise ValueError(
                    "Resolve unavailable or stale source series before updating alignment"
                )
            result = saved[source.series_id].result
            if result is None or not any(
                t.id_ == source.track_id and t.anchor_id == source.anchor_id for t in result.tracks
            ):
                raise ValueError("Selected trend disappeared; choose its replacement explicitly")
        return replace(
            table,
            rows=align_rows(table.sources, saved, table.rows),
            dependencies=tuple(
                (sid, dependency_stamp(saved[sid]))
                for sid in dict.fromkeys(s.series_id for s in table.sources)
            ),
        )

    def edit(
        self,
        table_id: str,
        *,
        name: str | None = None,
        selections: tuple[tuple[str, str], ...] | None = None,
        refresh_alignment: bool = False,
        column_order: tuple[str, ...] | None = None,
        plotted_columns: tuple[str, ...] | None = None,
    ) -> SetAnalyticalTable:
        """Edit presentation or explicitly update sources and positional correspondence."""
        old = self.query.get_analytical_table(table_id).state
        sources = old.sources
        if selections is not None:
            candidates = self._sources(selections)
            sources = tuple(
                next(
                    (
                        s
                        for s in old.sources
                        if (s.series_id, s.track_id) == (c.series_id, c.track_id)
                    ),
                    c,
                )
                for c in candidates
            )
        table = replace(
            old,
            name=old.name if name is None else name.strip(),
            sources=sources,
            rows=align_rows(
                sources,
                {
                    sid: self.query.get_post_processing_series(sid).series
                    for sid in self.query.get_post_processing_series_ids()
                },
                old.rows,
            )
            if selections is not None
            else old.rows,
            column_order=old.column_order if column_order is None else column_order,
            plotted_columns=old.plotted_columns if plotted_columns is None else plotted_columns,
            revision=new_revision(),
        )
        if refresh_alignment or selections is not None:
            # Old rows are input to alignment; temporary source changes are validated
            # only once all row references have been rebuilt below.
            table = self._align(table)
        return SetAnalyticalTable(table_id, table, old.revision)

    def remove(self, table_id: str) -> SetAnalyticalTable:
        """Delete a table without modifying its source series."""
        old = self.query.get_analytical_table(table_id).state
        return SetAnalyticalTable(table_id, None, old.revision)

    def save_metric(
        self,
        table_id: str,
        *,
        name: str,
        operation: str,
        tracks: tuple[str, ...],
        denominator: tuple[str, ...] = (),
        reference_row_id: str | None = None,
        metric_id: str | None = None,
    ) -> SetAnalyticalTable:
        """Save a fixed operation with explicit source sets and reference row."""
        old = self.query.get_analytical_table(table_id).state
        if metric_id is not None and not any(m.id_ == metric_id for m in old.metrics):
            raise ValueError("Unknown analysis column")
        source_ids = {s.id_ for s in old.sources}
        if not set((*tracks, *denominator)).issubset(source_ids):
            raise ValueError("Select operands from this table's sources")
        metric = TableMetric(
            metric_id or new_revision(),
            name.strip(),
            cast("MetricOperation", operation),
            tracks,
            denominator,
            reference_row_id,
        )
        metrics = tuple(metric if m.id_ == metric.id_ else m for m in old.metrics)
        if metric_id is None:
            metrics = (*metrics, metric)
        updated = replace(
            old,
            metrics=metrics,
            column_order=tuple(dict.fromkeys((*old.column_order, metric.id_))),
            plotted_columns=tuple(dict.fromkeys((*old.plotted_columns, metric.id_))),
            revision=new_revision(),
        )
        return SetAnalyticalTable(table_id, updated, old.revision)

    def remove_metric(self, table_id: str, metric_id: str) -> SetAnalyticalTable:
        """Remove a column and its saved ordering and graph selection."""
        old = self.query.get_analytical_table(table_id).state
        if not any(m.id_ == metric_id for m in old.metrics):
            raise ValueError("Unknown analysis column")
        updated = replace(
            old,
            metrics=tuple(m for m in old.metrics if m.id_ != metric_id),
            column_order=tuple(i for i in old.column_order if i != metric_id),
            plotted_columns=tuple(i for i in old.plotted_columns if i != metric_id),
            revision=new_revision(),
        )
        return SetAnalyticalTable(table_id, updated, old.revision)
