"""Structural validation shared by result application and persistence."""

from math import isfinite

from core.post_processing.matching import validate_constraints
from core.post_processing.types import MatchingResult


def validate_result(result: MatchingResult) -> None:
    """Reject broken path covers before accepting saved or worker results."""
    required = validate_constraints(result.series)
    forbidden = set(result.series.constraints.forbidden)
    selected = set()
    orders = {
        p.component_id: order for order, step in enumerate(result.series.steps) for p in step.peaks
    }
    covered = []
    track_ids = []
    if (
        not isfinite(result.objective)
        or type(result.candidate_links) is not int
        or result.candidate_links < 0
    ):
        raise ValueError("Invalid matching objective or candidate count")
    for track in result.tracks:
        ids = track.component_ids
        if not ids or track.anchor_id not in ids or track.id_ != f"track/{track.anchor_id}":
            raise ValueError("Invalid track identity or anchor")
        if len(track.links) != len(ids) - 1:
            raise ValueError("Track links do not cover its components")
        for index, link in enumerate(track.links):
            pair = link.source_id, link.target_id
            selected.add(pair)
            if pair in forbidden or link.manual != (pair in required):
                raise ValueError("Result links do not respect manual constraints")
            if (link.source_id, link.target_id) != (ids[index], ids[index + 1]):
                raise ValueError("Track link endpoints do not match its components")
            if orders.get(link.target_id, -2) != orders.get(link.source_id, -2) + 1:
                raise ValueError("Track links must connect adjacent steps")
            if not isfinite(link.cost) or any(not isfinite(cost) for _, cost in link.terms):
                raise ValueError("Nonfinite link cost")
            if link.cost_margin is not None and (
                not isfinite(link.cost_margin) or link.cost_margin < 0
            ):
                raise ValueError("Invalid ambiguity margin")
            if link.manual and (link.ambiguous or link.cost_margin is not None):
                raise ValueError("Manual links cannot carry an ambiguity estimate")
            if not link.manual and link.cost_margin is None:
                raise ValueError("Automatic links require an ambiguity estimate")
        covered.extend(ids)
        track_ids.append(track.id_)
    if (
        len(track_ids) != len(set(track_ids))
        or len(covered) != len(set(covered))
        or set(covered) != set(orders)
    ):
        raise ValueError("Tracks must cover every eligible peak exactly once")
    if not required.issubset(selected):
        raise ValueError("Result omits required manual links")
