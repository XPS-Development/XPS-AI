"""Undoable series editing and guarded application of immutable matching results."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from uuid import uuid4

from app.command.changes import SetPostProcessingSeries
from app.post_processing import PostProcessingInput, build_matching_input
from core.post_processing.state import PostProcessingSeries, new_revision
from core.post_processing.validation import validate_result

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.query_service import QueryService
    from core.post_processing.metrics import MetricDefinition
    from core.post_processing.types import LinkConstraints, MatchingResult, MatchingSettings


class PostProcessingUseCases:
    """Build changes; never mutate the document or retain live objects in jobs."""

    def __init__(self, query: QueryService) -> None:
        self._query = query

    def create(self, name: str, spectrum_ids: Sequence[str]) -> SetPostProcessingSeries:
        """Create a saved series in explicit supplied order, deduplicating selection."""
        ids = tuple(dict.fromkeys(spectrum_ids))
        available = set(self._query.get_all_spectra_ids())
        if any(sid not in available for sid in ids):
            raise ValueError("A new series can only contain existing spectra")
        series = PostProcessingSeries(uuid4().hex, name.strip(), ids)
        return SetPostProcessingSeries(series.id_, series)

    def edit(
        self,
        series_id: str,
        *,
        name: str | None = None,
        spectrum_ids: Sequence[str] | None = None,
        settings: MatchingSettings | None = None,
        constraints: LinkConstraints | None = None,
        metrics: tuple[MetricDefinition, ...] | None = None,
        column_order: tuple[str, ...] | None = None,
    ) -> SetPostProcessingSeries:
        """Edit configuration without discarding the previous result or unresolved operands."""
        old = self._query.get_post_processing_series(series_id).series
        updated = replace(
            old,
            name=old.name if name is None else name.strip(),
            spectrum_ids=old.spectrum_ids if spectrum_ids is None else tuple(spectrum_ids),
            settings=old.settings if settings is None else settings,
            constraints=old.constraints if constraints is None else constraints,
            metrics=old.metrics if metrics is None else metrics,
            column_order=old.column_order if column_order is None else column_order,
            revision=new_revision(),
        )
        return SetPostProcessingSeries(series_id, updated, old.revision)

    def remove(self, series_id: str) -> SetPostProcessingSeries:
        """Remove one series while retaining its full state for undo."""
        old = self._query.get_post_processing_series(series_id).series
        return SetPostProcessingSeries(series_id, None, old.revision)

    def name_track(
        self, series_id: str, track_id: str, name: str | None
    ) -> SetPostProcessingSeries:
        """Assign or clear a name by the track's component anchor."""
        old = self._query.get_post_processing_series(series_id).series
        if old.result is None:
            raise ValueError("Match the series before naming tracks")
        track = next((track for track in old.result.tracks if track.id_ == track_id), None)
        if track is None:
            raise ValueError("Unknown track")
        names = dict(old.track_names)
        if name is None or not name.strip():
            names.pop(track.anchor_id, None)
        else:
            names[track.anchor_id] = name.strip()
        updated = replace(old, track_names=tuple(names.items()), revision=new_revision())
        return SetPostProcessingSeries(series_id, updated, old.revision)

    def prepare(self, series_id: str) -> PostProcessingInput:
        """Build immutable worker input and capture the current version guards."""
        saved = self._query.get_post_processing_series(series_id).series
        return PostProcessingInput(
            series_id,
            saved.revision,
            self._query.document_generation,
            self._query.get_post_processing_fingerprint(saved),
            build_matching_input(self._query, saved),
        )

    def apply_result(
        self, job: PostProcessingInput, result: MatchingResult
    ) -> SetPostProcessingSeries:
        """Accept a current result atomically, preserving track identities by anchors."""
        old = self._query.get_post_processing_series(job.series_id).series
        if (
            job.document_generation != self._query.document_generation
            or job.revision != old.revision
            or job.fingerprint != self._query.get_post_processing_fingerprint(old)
        ):
            raise ValueError("Matching input changed; discard the outdated result")
        if result.series != job.matching:
            raise ValueError("Matching result does not belong to the supplied job")
        validate_result(result)
        old_anchors = (
            {}
            if old.result is None
            else {track.anchor_id: track.id_ for track in old.result.tracks}
        )
        named = set(dict(old.track_names))
        observed = {cid for track in result.tracks for cid in track.component_ids}
        if named - observed:
            raise ValueError("Named track anchor disappeared; resolve the track name conflict")
        tracks = []
        for track in result.tracks:
            named_anchors = named.intersection(track.component_ids)
            if len(named_anchors) > 1:
                raise ValueError("Several named anchors merged; resolve the track name conflict")
            anchor = next(iter(named_anchors), None)
            if anchor is None:
                anchor = next(
                    (cid for cid in track.component_ids if cid in old_anchors), track.anchor_id
                )
            tracks.append(replace(track, id_=f"track/{anchor}", anchor_id=anchor))
        updated = replace(
            old,
            result=replace(result, tracks=tuple(tracks)),
            input_fingerprint=job.fingerprint,
            revision=new_revision(),
        )
        return SetPostProcessingSeries(old.id_, updated, old.revision)
