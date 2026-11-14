"""Human-readable trend labels shared by projections and exports."""

from core.post_processing.state import PostProcessingSeries


def trend_label(series: PostProcessingSeries, track_id: str) -> str:
    """Return a saved name or a short numbered label, without exposing identities."""
    if series.result is not None:
        for index, track in enumerate(series.result.tracks, 1):
            if track.id_ == track_id:
                return dict(series.track_names).get(track.anchor_id, f"Trend {index}")
    return "Unavailable trend"
