"""
Core protocol types for structural typing across the application.

Defines minimal interfaces (ParameterLike, BaseLike, ComponentLike, RegionLike,
SpectrumLike) that domain objects, DTOs, and library API consumers can use
for duck typing without tight coupling to concrete implementations.
"""

from collections.abc import Mapping
from typing import Final, Literal, Protocol

from numpy.typing import NDArray

from core.math_models import ParametricModelLike


class ParameterLike(Protocol):
    """
    Protocol describing a parameter-like object.

    Defines the minimal attribute set required for parameter
    normalization, denormalization, and DTO construction.
    """

    name: Final[str]
    value: Final[float]
    lower: Final[float]
    upper: Final[float]
    vary: Final[bool]
    expr: Final[str | None]


class BaseLike(Protocol):
    """
    Protocol for common identity and normalization metadata.

    Shared by all core object projections (region, spectrum, component).
    """

    id_: Final[str]
    parent_id: Final[str | None]
    normalized: Final[bool]


class ComponentLike(BaseLike, Protocol):
    """
    Protocol for component-like objects with parameters and model metadata.

    Satisfied by ComponentDTO and domain Component objects.
    """

    parameters: Final[Mapping[str, ParameterLike]]
    model: Final[ParametricModelLike]
    kind: Final[Literal["peak", "background"]]


class RegionLike(BaseLike, Protocol):
    """
    Protocol for region-like objects with numerical data.

    Satisfied by RegionDTO and similar projections.
    """

    x: Final[NDArray]
    y: Final[NDArray]


class SpectrumLike(BaseLike, Protocol):
    """
    Protocol for spectrum-like objects with numerical data.

    Satisfied by SpectrumDTO and similar projections.
    """

    x: Final[NDArray]
    y: Final[NDArray]
