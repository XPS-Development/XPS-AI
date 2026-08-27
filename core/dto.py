"""Immutable DTO projections of core domain objects."""

from dataclasses import dataclass
from typing import Literal

from numpy.typing import NDArray

from core.math_models import ParametricModelLike


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
