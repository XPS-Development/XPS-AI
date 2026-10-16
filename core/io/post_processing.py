"""JSON codec for the independent post-processing section of a document."""

import json
from dataclasses import asdict
from typing import Any, cast

from core.post_processing.metrics import MetricDefinition
from core.post_processing.state import PostProcessingSeries
from core.post_processing.types import (
    ExcludedComponent,
    LinkConstraints,
    MatchingResult,
    MatchingSeries,
    MatchingSettings,
    PeakDescriptor,
    PeakLink,
    PeakTrack,
    SeriesStep,
)
from core.post_processing.validation import validate_result


def _record(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Expected a JSON object in post_processing")
    return cast(dict[str, Any], value)


def _items(value: object) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array in post_processing")
    return value


def _strings(value: object) -> tuple[str, ...]:
    items = _items(value)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError("Expected nonempty string IDs in post_processing")
    return tuple(items)


def _pairs(value: object) -> tuple[tuple[str, str], ...]:
    result = []
    for item in _items(value):
        pair = _strings(item)
        if len(pair) != 2:
            raise ValueError("Expected a pair in post_processing")
        result.append((pair[0], pair[1]))
    return tuple(result)


def _constraints(value: object) -> LinkConstraints:
    data = _record(value)
    return LinkConstraints(_pairs(data["required"]), _pairs(data["forbidden"]))


def _matching_series(value: object) -> MatchingSeries:
    data = _record(value)
    steps = []
    for item in _items(data["steps"]):
        step = _record(item)
        if type(step["available"]) is not bool:
            raise ValueError("Step availability must be boolean")
        steps.append(
            SeriesStep(
                spectrum_id=step["spectrum_id"],
                peaks=tuple(PeakDescriptor(**_record(peak)) for peak in _items(step["peaks"])),
                excluded=tuple(ExcludedComponent(**_record(c)) for c in _items(step["excluded"])),
                available=step["available"],
            )
        )
    return MatchingSeries(
        data["id_"],
        tuple(steps),
        MatchingSettings(**_record(data["settings"])),
        _constraints(data["constraints"]),
    )


def _result(value: object) -> MatchingResult:
    data = _record(value)
    series = _matching_series(data["series"])
    tracks = []
    for item in _items(data["tracks"]):
        track = _record(item)
        links = []
        for raw_link in _items(track["links"]):
            link = dict(_record(raw_link))
            link["terms"] = tuple((name, cost) for name, cost in _items(link["terms"]))
            link["missing_features"] = _strings(link["missing_features"])
            links.append(PeakLink(**link))
        tracks.append(
            PeakTrack(
                track["id_"], track["anchor_id"], _strings(track["component_ids"]), tuple(links)
            )
        )
    result = MatchingResult(series, tuple(tracks), data["objective"], data["candidate_links"])
    validate_result(result)
    return result


def serialize_series(series: PostProcessingSeries) -> dict[str, Any]:
    """Encode every saved field, retaining stale results and unresolved metrics."""
    if series.result is not None:
        validate_result(series.result)
    return json.loads(json.dumps(asdict(series), allow_nan=False))


def deserialize_series(value: object) -> PostProcessingSeries:
    """Decode one saved series and validate its immutable result structure."""
    try:
        data = dict(_record(value))
        data["spectrum_ids"] = _strings(data["spectrum_ids"])
        data["settings"] = MatchingSettings(**_record(data["settings"]))
        data["constraints"] = _constraints(data["constraints"])
        data["result"] = None if data["result"] is None else _result(data["result"])
        data["track_names"] = _pairs(data["track_names"])
        metrics = []
        for raw in _items(data["metrics"]):
            metric = dict(_record(raw))
            metric["tracks"] = _strings(metric["tracks"])
            metric["denominator"] = _strings(metric["denominator"])
            metrics.append(MetricDefinition(**metric))
        data["metrics"] = tuple(metrics)
        data["column_order"] = _strings(data["column_order"])
        return PostProcessingSeries(**data)
    except (TypeError, KeyError, OverflowError) as error:
        raise ValueError(f"Invalid post_processing series: {error}") from error


def deserialize_section(value: object) -> tuple[PostProcessingSeries, ...]:
    """Decode the complete section before mutating a target document."""
    series = tuple(deserialize_series(item) for item in _items(value))
    ids = [item.id_ for item in series]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate post-processing series IDs")
    return series
