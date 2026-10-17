"""Immutable DTO projections of core domain objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from core.math_models import ParametricModelLike
    from core.post_processing.analysis import AnalyticalTableState
    from core.post_processing.state import PostProcessingSeries


@dataclass(frozen=True)
class ParameterDTO:
    """
    Immutable data transfer object representing a single model parameter.

    Used for normalized and denormalized parameter exchange between
    services without mutating domain state.
    """

    name: str
    value: float
    lower: float
    upper: float
    vary: bool
    expr: str | None


@dataclass(frozen=True)
class BaseDTO:
    """
    Base immutable projection of a core domain object.

    Contains common identity and normalization metadata shared
    by all DTO projections.
    """

    id_: str
    parent_id: str | None
    normalized: bool


@dataclass(frozen=True)
class ComponentDTO(BaseDTO):
    """
    Immutable projection of a spectral component.

    Encapsulates model metadata and a snapshot of component
    parameters in either normalized or denormalized form.
    """

    parameters: dict[str, ParameterDTO]
    model: ParametricModelLike
    kind: Literal["peak", "background"]
    name: str | None = None


@dataclass(frozen=True)
class RegionDTO(BaseDTO):
    """
    Immutable projection of region numerical data.

    References sliced views of the parent spectrum arrays and
    does not own data independently.
    """

    x: NDArray
    y: NDArray


@dataclass(frozen=True)
class SpectrumDTO(BaseDTO):
    """
    Immutable projection of spectrum numerical data.

    Provides access to raw or normalized spectrum arrays without
    exposing mutable domain objects.
    """

    x: NDArray
    y: NDArray


@dataclass(frozen=True)
class PostProcessingSeriesDTO:
    """Immutable series projection with live availability and snapshot status."""

    series: PostProcessingSeries
    stale: bool
    unavailable_spectrum_ids: tuple[str, ...]
    conflicts: tuple[str, ...]

    @property
    def exportable(self) -> bool:
        """Return whether the current result can be exported normally."""
        return not self.stale and not self.unavailable_spectrum_ids and not self.conflicts


@dataclass(frozen=True)
class SourceLocation:
    """Original component behind one value, including cross-series calculations."""

    series_id: str
    track_id: str
    spectrum_id: str
    region_id: str
    component_id: str


@dataclass(frozen=True)
class AnalysisCell:
    """Full precision table cell with calculation status and missing-value reason."""

    value: str | float | int | None
    status: str = "ok"
    reason: str | None = None
    sources: tuple[SourceLocation, ...] = ()


@dataclass(frozen=True)
class AnalysisColumn:
    """Stable column identity and display/export metadata."""

    id_: str
    label: str
    unit: str = ""
    track_id: str | None = None
    anchor_id: str | None = None
    quantity: str = ""


@dataclass(frozen=True)
class AnalysisRow:
    """One table row with stable source navigation, independent of sorting."""

    step: int
    spectrum_id: str
    cells: tuple[AnalysisCell, ...]
    region_id: str | None = None
    component_id: str | None = None
    id_: str = ""


@dataclass(frozen=True)
class AnalysisTable:
    """Shared immutable projection for the table, trends and CSV."""

    columns: tuple[AnalysisColumn, ...]
    rows: tuple[AnalysisRow, ...]


@dataclass(frozen=True)
class AnalyticalTableDTO:
    """Independent table projection with dependency checks and nonblocking warnings."""

    state: AnalyticalTableState
    table: AnalysisTable
    stale: bool
    conflicts: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def exportable(self) -> bool:
        """Permit gaps and unequal lengths, but reject unresolved dependencies."""
        return bool(self.state.sources) and not self.stale and not self.conflicts
