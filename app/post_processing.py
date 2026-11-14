"""Read-only snapshots and projections for document post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.dto import PostProcessingSeriesDTO
from core.post_processing import MatchingSeries, SeriesStep, step_from_components
from core.post_processing.matching import validate_constraints

if TYPE_CHECKING:
    from app.query_service import QueryService
    from core.post_processing.state import PostProcessingSeries


@dataclass(frozen=True)
class PostProcessingInput:
    """Immutable matching job with guards against stale and replaced documents."""

    series_id: str
    revision: str
    document_generation: str
    fingerprint: str
    matching: MatchingSeries


def build_matching_input(query: QueryService, series: PostProcessingSeries) -> MatchingSeries:
    """Collect physical DTOs across all regions, retaining unavailable steps."""
    available = set(query.get_all_spectra_ids())
    steps = []
    for spectrum_id in series.spectrum_ids:
        if spectrum_id not in available:
            steps.append(SeriesStep(spectrum_id, available=False))
            continue
        _, regions = query.get_spectrum_dto_repr(spectrum_id, normalized=False)
        steps.append(
            step_from_components(spectrum_id, (c for _, components in regions for c in components))
        )
    return MatchingSeries(series.id_, tuple(steps), series.settings, series.constraints)


def project_series(query: QueryService, series: PostProcessingSeries) -> PostProcessingSeriesDTO:
    """Expose current availability, stale status and unresolved references."""
    available = set(query.get_all_spectra_ids())
    unavailable = tuple(sid for sid in series.spectrum_ids if sid not in available)
    fingerprint = query.get_post_processing_fingerprint(series)
    stale = series.result is None or fingerprint != series.input_fingerprint
    conflicts = []
    if series.constraints.required or series.constraints.forbidden:
        try:
            validate_constraints(build_matching_input(query, series))
        except ValueError as error:
            conflicts.append(f"constraint_conflict:{error}")
    if series.result is not None:
        anchors = {track.anchor_id for track in series.result.tracks}
        for anchor, _ in series.track_names:
            if anchor not in anchors:
                conflicts.append(f"missing_track_anchor:{anchor}")
    return PostProcessingSeriesDTO(series, stale, unavailable, tuple(conflicts))


@dataclass(frozen=True)
class MetricOperationOption:
    """Presentation metadata for a supported operation's explicit operands."""

    id_: str
    label: str
    operands: str
    reference: bool = False


METRIC_OPERATIONS = (
    MetricOperationOption("position", "Position (eV)", "one"),
    MetricOperationOption("area", "Area", "one"),
    MetricOperationOption("sig", "sig (eV)", "one"),
    MetricOperationOption("spectrum_area_fraction", "Rel. area (%)", "one"),
    MetricOperationOption("position_difference", "Position A - position B (eV)", "two"),
    MetricOperationOption(
        "position_shift", "Position - position at reference step (eV)", "one", True
    ),
    MetricOperationOption("area_sum", "Sum of selected areas", "sum"),
    MetricOperationOption("area_ratio", "Area A / area B", "two"),
    MetricOperationOption(
        "area_sum_ratio", "Sum of numerator areas / sum of denominator areas", "two_sets"
    ),
    MetricOperationOption("area_fraction", "Rel. area of selected peaks (%)", "fraction"),
)
