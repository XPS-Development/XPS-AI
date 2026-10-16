"""Persistent series definitions, separate from calculated values and live spectra."""

from dataclasses import dataclass, field
from uuid import uuid4

from core.post_processing.metrics import MetricDefinition
from core.post_processing.types import (
    LinkConstraints,
    MatchingResult,
    MatchingSettings,
)


def new_revision() -> str:
    """Return an opaque revision token for a saved series edit."""
    return uuid4().hex


@dataclass(frozen=True)
class PostProcessingSeries:
    """Saved series configuration and last result, retained even when stale.

    Track names are keyed by component anchor IDs. Metric definitions refer to
    stable track IDs. Availability and staleness are derived from the live
    document, so undo restores them without modifying the saved snapshot.
    """

    id_: str
    name: str
    spectrum_ids: tuple[str, ...]
    settings: MatchingSettings = field(default_factory=MatchingSettings)
    constraints: LinkConstraints = field(default_factory=LinkConstraints)
    result: MatchingResult | None = None
    input_fingerprint: str | None = None
    track_names: tuple[tuple[str, str], ...] = ()
    metrics: tuple[MetricDefinition, ...] = ()
    column_order: tuple[str, ...] = ()
    revision: str = field(default_factory=new_revision)

    def __post_init__(self) -> None:
        """Validate persistent identities without discarding unresolved operands."""
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.id_, self.name, self.revision)
        ):
            raise ValueError("Series ID, name and revision must be nonempty")
        if not self.spectrum_ids or any(
            not isinstance(sid, str) or not sid for sid in self.spectrum_ids
        ):
            raise ValueError("A series requires at least one spectrum ID")
        if len(self.spectrum_ids) != len(set(self.spectrum_ids)):
            raise ValueError("Spectrum IDs must be unique in a series")
        for values, label in (
            ([anchor for anchor, _ in self.track_names], "Track name anchors"),
            ([metric.id_ for metric in self.metrics], "Metric IDs"),
            (list(self.column_order), "Column IDs"),
        ):
            if any(not value for value in values) or len(values) != len(set(values)):
                raise ValueError(f"{label} must be nonempty and unique")
        if any(not name.strip() for _, name in self.track_names):
            raise ValueError("Track names must be nonempty")
        if (self.result is None) != (self.input_fingerprint is None):
            raise ValueError("A saved result and its input fingerprint must be stored together")
        if self.input_fingerprint is not None and (
            not isinstance(self.input_fingerprint, str) or not self.input_fingerprint
        ):
            raise ValueError("Input fingerprint must be a nonempty string")
        if self.result is not None and self.result.series.id_ != self.id_:
            raise ValueError("Result must belong to the saved series")
