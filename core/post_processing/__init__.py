"""Pure spectrum-series matching and numerical post-processing."""

from core.post_processing.adapter import step_from_components
from core.post_processing.matching import match_series
from core.post_processing.metrics import MetricDefinition, MetricValue, calculate_metrics
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

__all__ = [
    "ExcludedComponent",
    "LinkConstraints",
    "MatchingResult",
    "MatchingSeries",
    "MatchingSettings",
    "MetricDefinition",
    "MetricValue",
    "PeakDescriptor",
    "PeakLink",
    "PeakTrack",
    "SeriesStep",
    "calculate_metrics",
    "match_series",
    "step_from_components",
]
