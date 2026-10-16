"""Shared numerical values for future tables, trends and exports."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from core.post_processing.types import MatchingResult, PeakDescriptor

MetricOperation = Literal[
    "position",
    "area",
    "fwhm",
    "sig",
    "spectrum_area_fraction",
    "position_difference",
    "position_shift",
    "area_sum",
    "area_ratio",
    "area_sum_ratio",
    "area_fraction",
]
ValueStatus = Literal["ok", "ambiguous", "missing"]


@dataclass(frozen=True)
class MetricDefinition:
    """Persistable operation using stable track and reference spectrum IDs.

    ``tracks`` supplies the ordered operands or the numerator set; denominator
    supplies the denominator set. Repeated IDs within a sum are counted once.
    Fractions are percentages; ratios are dimensionless.
    """

    id_: str
    name: str
    operation: MetricOperation
    tracks: tuple[str, ...]
    denominator: tuple[str, ...] = ()
    reference_spectrum_id: str | None = None

    def __post_init__(self) -> None:
        """Validate operation arity, explicit sets and reference requirements."""
        if not self.id_ or not self.name:
            raise ValueError("Metric ID and name must be nonempty")
        counts = {
            "position": 1,
            "area": 1,
            "fwhm": 1,
            "sig": 1,
            "spectrum_area_fraction": 1,
            "position_difference": 2,
            "position_shift": 1,
            "area_ratio": 2,
            "area_fraction": 1,
        }
        if self.operation in counts:
            if len(self.tracks) != counts[self.operation]:
                raise ValueError(f"{self.operation} requires {counts[self.operation]} operands")
        elif self.operation in ("area_sum", "area_sum_ratio"):
            if not self.tracks:
                raise ValueError("Area sums require an explicit nonempty track set")
        else:
            raise ValueError(f"Unknown metric operation: {self.operation}")
        if self.operation in ("area_sum_ratio", "area_fraction"):
            if not self.denominator:
                raise ValueError("An explicit denominator track set is required")
        elif self.denominator:
            raise ValueError("This operation does not accept a denominator set")
        if self.operation == "area_fraction" and self.tracks[0] not in self.denominator:
            raise ValueError("The selected track must belong to the denominator set")
        if self.operation == "position_shift":
            if not self.reference_spectrum_id:
                raise ValueError("Position shift requires an explicit reference spectrum ID")
        elif self.reference_spectrum_id is not None:
            raise ValueError("Only position shift accepts a reference spectrum")

    @property
    def unit(self) -> str:
        """Return the unit independently of whether values are available."""
        if self.operation in ("area_fraction", "spectrum_area_fraction"):
            return "%"
        if self.operation in ("area_ratio", "area_sum_ratio"):
            return "1"
        if self.operation in ("area", "area_sum"):
            return "area"
        return "eV"


@dataclass(frozen=True)
class MetricValue:
    """Full precision cell; absent values always carry a machine-readable reason."""

    metric_id: str
    spectrum_id: str
    value: float | None
    status: ValueStatus
    reason: str | None = None


class _MissingValue(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


class _Evaluator:
    def __init__(self, result: MatchingResult) -> None:
        self.steps = {step.spectrum_id: step for step in result.series.steps}
        self.tracks = {track.id_: track for track in result.tracks}
        self.peaks = {
            (step.spectrum_id, peak.component_id): peak
            for step in result.series.steps
            for peak in step.peaks
        }
        self.ambiguous = False

    def peak(self, track_id: str, spectrum_id: str) -> PeakDescriptor:
        track = self.tracks.get(track_id)
        if track is None:
            raise _MissingValue("operand_selection_required")
        step = self.steps.get(spectrum_id)
        if step is None:
            raise _MissingValue("reference_selection_required")
        if not step.available:
            raise _MissingValue("unavailable_spectrum")
        for component_id in track.component_ids:
            peak = self.peaks.get((spectrum_id, component_id))
            if peak is not None:
                self.ambiguous |= track.ambiguous
                return peak
        raise _MissingValue("missing_peak")

    def number(self, value: float | None) -> float:
        if value is None:
            raise _MissingValue("missing_feature")
        if not isfinite(value):
            raise _MissingValue("nonfinite_value")
        return value

    def area_sum(self, tracks: tuple[str, ...], spectrum_id: str) -> float:
        return self.number(
            sum(self.number(self.peak(t, spectrum_id).area) for t in dict.fromkeys(tracks))
        )

    def ratio(self, numerator: float, denominator: float) -> float:
        if denominator == 0:
            raise _MissingValue("zero_denominator")
        return self.number(numerator / denominator)

    def evaluate(self, definition: MetricDefinition, spectrum_id: str) -> float:
        op = definition.operation
        tracks = definition.tracks
        if op == "area_sum":
            return self.area_sum(tracks, spectrum_id)
        if op == "area_sum_ratio":
            return self.ratio(
                self.area_sum(tracks, spectrum_id),
                self.area_sum(definition.denominator, spectrum_id),
            )
        peak = self.peak(tracks[0], spectrum_id)
        if op in ("position", "area", "fwhm", "sig"):
            return self.number(getattr(peak, op))
        if op == "position_difference":
            return peak.position - self.peak(tracks[1], spectrum_id).position
        if op == "position_shift":
            # Validated definitions always have a reference ID; never substitute a step.
            reference = self.peak(tracks[0], definition.reference_spectrum_id or "")
            return peak.position - reference.position
        if op == "area_ratio":
            return self.ratio(
                self.number(peak.area), self.number(self.peak(tracks[1], spectrum_id).area)
            )
        if op == "area_fraction":
            return 100 * self.ratio(
                self.number(peak.area), self.area_sum(definition.denominator, spectrum_id)
            )
        # The explicitly named source fraction uses all eligible peaks of this spectrum.
        total = self.number(sum(self.number(p.area) for p in self.steps[spectrum_id].peaks))
        return 100 * self.ratio(self.number(peak.area), total)


def calculate_metrics(
    result: MatchingResult, definitions: tuple[MetricDefinition, ...]
) -> tuple[MetricValue, ...]:
    """Evaluate every definition on every supplied step, preserving order and gaps.

    Missing operands never become zero or partial sums. Ambiguity is propagated
    conservatively from any track used by a calculation, including its reference.
    Track IDs that no longer exist require explicit operand selection.
    """
    ids = [definition.id_ for definition in definitions]
    if len(ids) != len(set(ids)):
        raise ValueError("Metric IDs must be unique")
    evaluator = _Evaluator(result)
    values = []
    for step in result.series.steps:
        for definition in definitions:
            evaluator.ambiguous = False
            try:
                if not step.available:
                    raise _MissingValue("unavailable_spectrum")
                value = evaluator.number(evaluator.evaluate(definition, step.spectrum_id))
            except _MissingValue as error:
                values.append(
                    MetricValue(definition.id_, step.spectrum_id, None, "missing", error.reason)
                )
            else:
                status = "ambiguous" if evaluator.ambiguous else "ok"
                values.append(MetricValue(definition.id_, step.spectrum_id, value, status))
    return tuple(values)
