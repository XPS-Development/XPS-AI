"""Immutable inputs and results for spectrum-series post-processing."""

from dataclasses import dataclass, fields
from math import isfinite


@dataclass(frozen=True)
class MatchingSettings:
    """Active continuous-matching parameters, in physical energy units."""

    energy_scale: float = 0.75
    energy_gate: float = 4.0
    width_scale: float = 0.3
    shape_scale: float = 0.2
    energy_weight: float = 1.0
    width_weight: float = 0.0
    shape_weight: float = 0.0
    start_cost: float = 10.0
    end_cost: float = 10.0
    ambiguity_threshold: float = 0.5
    area_weight: float = 0.1
    rank_weight: float = 0.1

    def __post_init__(self) -> None:
        """Reject invalid scales and costs before a worker is started."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative")
        for name in ("energy_scale", "energy_gate", "width_scale", "shape_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PeakDescriptor:
    """Physical peak values; region identity is provenance, never a match gate."""

    component_id: str
    region_id: str
    model_name: str
    position: float
    area: float | None = None
    fwhm: float | None = None
    sig: float | None = None
    shape_gl: float | None = None

    def __post_init__(self) -> None:
        """Keep invalid numerical inputs out of assignment and logarithms."""
        if any(
            not isinstance(value, str) or not value
            for value in (self.component_id, self.region_id, self.model_name)
        ):
            raise ValueError("Peak identities and model name must be nonempty")
        if not isfinite(self.position):
            raise ValueError("Peak position must be finite")
        for name in ("area", "fwhm", "sig", "shape_gl"):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite or absent")
        if self.fwhm is not None and self.fwhm <= 0:
            raise ValueError("FWHM must be positive or absent")


@dataclass(frozen=True)
class ExcludedComponent:
    """Component omitted from a step, with an explicit machine-readable reason."""

    component_id: str
    region_id: str
    reason: str


@dataclass(frozen=True)
class SeriesStep:
    """One supplied spectrum, including an empty or unavailable step."""

    spectrum_id: str
    peaks: tuple[PeakDescriptor, ...] = ()
    excluded: tuple[ExcludedComponent, ...] = ()
    available: bool = True


@dataclass(frozen=True)
class LinkConstraints:
    """Directed adjacent component pairs fixed or forbidden by the user."""

    required: tuple[tuple[str, str], ...] = ()
    forbidden: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class MatchingSeries:
    """Ordered immutable worker input; no live models, arrays or domain objects."""

    id_: str
    steps: tuple[SeriesStep, ...]
    settings: MatchingSettings = MatchingSettings()
    constraints: LinkConstraints = LinkConstraints()

    def __post_init__(self) -> None:
        """Validate identities and unavailable-step invariants."""
        if not isinstance(self.id_, str) or not self.id_:
            raise ValueError("Series ID must be nonempty")
        spectra = [step.spectrum_id for step in self.steps]
        if any(not isinstance(sid, str) or not sid for sid in spectra) or len(spectra) != len(
            set(spectra)
        ):
            raise ValueError("Spectrum IDs must be nonempty and unique in a series")
        components = [
            peak.component_id for step in self.steps for peak in (*step.peaks, *step.excluded)
        ]
        if len(components) != len(set(components)):
            raise ValueError("Component IDs must be unique in a series")
        if any(not step.available and step.peaks for step in self.steps):
            raise ValueError("Unavailable steps cannot contain eligible peaks")


@dataclass(frozen=True)
class PeakLink:
    """Selected adjacent link; manual links have no ambiguity estimate."""

    source_id: str
    target_id: str
    cost: float
    terms: tuple[tuple[str, float], ...]
    missing_features: tuple[str, ...] = ()
    manual: bool = False
    cost_margin: float | None = None
    ambiguous: bool = False


@dataclass(frozen=True)
class PeakTrack:
    """Continuous component chain, identified by its first component anchor."""

    id_: str
    anchor_id: str
    component_ids: tuple[str, ...]
    links: tuple[PeakLink, ...]

    @property
    def ambiguous(self) -> bool:
        """Return whether any automatic link in the chain is ambiguous."""
        return any(link.ambiguous for link in self.links)


@dataclass(frozen=True)
class MatchingResult:
    """Path cover and its original immutable input, including empty steps."""

    series: MatchingSeries
    tracks: tuple[PeakTrack, ...]
    objective: float
    candidate_links: int
