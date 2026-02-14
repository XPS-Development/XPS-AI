"""
Core protocol types for structural typing across the application.

Defines minimal interfaces (ParameterLike, BaseLike, ComponentLike, RegionLike,
SpectrumLike) that domain objects, DTOs, and library API consumers can use
for duck typing without tight coupling to concrete implementations.
"""

from typing import Literal, Protocol

from numpy.typing import NDArray

from core.math_models import ParametricModelLike


class ParameterLike(Protocol):
    """
    Protocol describing a parameter-like object.

    Defines the minimal attribute set required for parameter
    normalization, denormalization, and DTO construction.
    """

    name: str
    value: float
    lower: float
    upper: float
    vary: bool
    expr: str | None


class BaseLike(Protocol):
    """
    Protocol for common identity and normalization metadata.

    Shared by all core object projections (region, spectrum, component).
    """

    id_: str
    parent_id: str | None
    normalized: bool


class ComponentLike(BaseLike, Protocol):
    """
    Protocol for component-like objects with parameters and model metadata.

    Satisfied by ComponentDTO and domain Component objects.
    """

    parameters: dict[str, ParameterLike]
    model: ParametricModelLike
    kind: Literal["peak", "background"]


class RegionLike(BaseLike, Protocol):
    """
    Protocol for region-like objects with numerical data.

    Satisfied by RegionDTO and similar projections.
    """

    x: NDArray
    y: NDArray


class SpectrumLike(BaseLike, Protocol):
    """
    Protocol for spectrum-like objects with numerical data.

    Satisfied by SpectrumDTO and similar projections.
    """

    x: NDArray
    y: NDArray
