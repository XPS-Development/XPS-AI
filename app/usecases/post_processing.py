"""Undoable series editing and guarded application of immutable matching results."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from uuid import uuid4

from app.command.changes import SetPostProcessingSeries
from app.post_processing import PostProcessingInput, build_matching_input
from core.post_processing.matching import match_series
from core.post_processing.state import PostProcessingSeries, new_revision
from core.post_processing.validation import validate_result

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.query_service import QueryService
    from core.post_processing.types import LinkConstraints, MatchingResult, MatchingSettings


class PostProcessingUseCases:
    """Build changes; never mutate the document or retain live objects in jobs."""

    def __init__(self, query: QueryService) -> None:
        self._query = query

    def create(self, name: str, spectrum_ids: Sequence[str]) -> SetPostProcessingSeries:
        """Create and match an ordered series as one undoable change."""
        ids = tuple(dict.fromkeys(spectrum_ids))
        available = set(self._query.get_all_spectra_ids())
        if any(sid not in available for sid in ids):
            raise ValueError("A new series can only contain existing spectra")
        series = PostProcessingSeries(uuid4().hex, name.strip(), ids)
        result = match_series(build_matching_input(self._query, series))
        series = replace(
            series,
            result=result,
            input_fingerprint=self._query.get_post_processing_fingerprint(series),
        )
        return SetPostProcessingSeries(series.id_, series)

    def edit(
        self,
        series_id: str,
        *,
        name: str | None = None,
        spectrum_ids: Sequence[str] | None = None,
        settings: MatchingSettings | None = None,
        constraints: LinkConstraints | None = None,
    ) -> SetPostProcessingSeries:
        """Edit configuration without discarding the previous result or unresolved operands."""
        old = self._query.get_post_processing_series(series_id).series
        updated = replace(
            old,
            name=old.name if name is None else name.strip(),
            spectrum_ids=old.spectrum_ids if spectrum_ids is None else tuple(spectrum_ids),
            settings=old.settings if settings is None else settings,
            constraints=old.constraints if constraints is None else constraints,
            revision=new_revision(),
        )
        return SetPostProcessingSeries(series_id, updated, old.revision)

    def remove(self, series_id: str) -> SetPostProcessingSeries:
        """Remove one series while retaining its full state for undo."""
        old = self._query.get_post_processing_series(series_id).series
        return SetPostProcessingSeries(series_id, None, old.revision)

    def run(self, series_id: str) -> SetPostProcessingSeries:
        """Match an immutable snapshot synchronously and build one result command."""
        snapshot = self.prepare(series_id)
        return self.apply_result(snapshot, match_series(snapshot.matching))

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
        """Build an immutable synchronous input snapshot with current version guards."""
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
        result = self._preserve_tracks(old, result)
        updated = replace(
            old,
            result=result,
            input_fingerprint=job.fingerprint,
            revision=new_revision(),
        )
        return SetPostProcessingSeries(old.id_, updated, old.revision)

    def _preserve_tracks(self, old: PostProcessingSeries, result: MatchingResult) -> MatchingResult:
        """Preserve named and selected anchors, rejecting implicit identity changes."""
        old_anchors = (
            {}
            if old.result is None
            else {track.anchor_id: track.id_ for track in old.result.tracks}
        )
        named = set(dict(old.track_names)) | self._query.get_used_trend_anchors(old.id_)
        observed = {cid for track in result.tracks for cid in track.component_ids}
        if named - observed:
            raise ValueError(
                "Named or selected trend anchor disappeared; resolve the track name conflict"
            )
        tracks = []
        for track in result.tracks:
            named_anchors = named.intersection(track.component_ids)
            if len(named_anchors) > 1:
                raise ValueError(
                    "Several named or selected anchors merged; resolve the track name conflict"
                )
            anchor = next(iter(named_anchors), None)
            if anchor is None:
                anchor = next(
                    (cid for cid in track.component_ids if cid in old_anchors), track.anchor_id
                )
            tracks.append(replace(track, id_=f"track/{anchor}", anchor_id=anchor))
        return replace(result, tracks=tuple(tracks))

    def set_link(
        self, series_id: str, source_id: str, target_id: str, linked: bool
    ) -> SetPostProcessingSeries:
        """Assign or break an adjacent link and recalculate in the same undo step."""
        from core.post_processing.types import LinkConstraints

        old = self._query.get_post_processing_series(series_id).series
        required = set(old.constraints.required)
        forbidden = set(old.constraints.forbidden)
        pair = source_id, target_id
        if linked:
            if any(
                (a == source_id and b != target_id) or (b == target_id and a != source_id)
                for a, b in required
            ):
                raise ValueError("Conflicting manual predecessor or successor; break it first")
            required.add(pair)
            forbidden.discard(pair)
        else:
            required.discard(pair)
            forbidden.add(pair)
        edited = replace(
            old, constraints=LinkConstraints(tuple(sorted(required)), tuple(sorted(forbidden)))
        )
        matching = build_matching_input(self._query, edited)
        result = match_series(matching)
        result = self._preserve_tracks(old, result)
        updated = replace(
            edited,
            result=result,
            input_fingerprint=self._query.get_post_processing_fingerprint(edited),
            revision=new_revision(),
        )
        return SetPostProcessingSeries(series_id, updated, old.revision)
