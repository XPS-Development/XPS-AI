"""Base classes, protocols, and parameter schemas for parametric models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from .normalization import NormalizationContext, ParameterNormalizationPolicy

T = TypeVar("T", bound=np.floating)


@dataclass(frozen=True)
class ParameterSpec:
    """Schema entry describing one model parameter and its defaults."""

    name: str
    default: float
    lower: float = -np.inf
    upper: float = np.inf
    vary: bool = True
    expr: str | None = None


class ParametricModelLike(Protocol[T]):
    """Structural interface for a named parametric model instance."""

    name: ClassVar[str]
    parameter_schema: ClassVar[tuple[ParameterSpec, ...]]
    independent_vars: ClassVar[tuple[str, ...]]
    normalization_target_parameters: ClassVar[tuple[str, ...]]
    use_offset: ClassVar[bool]
    use_scale: ClassVar[bool]

    @staticmethod
    def evaluate(x: NDArray[T], y: NDArray[T] | None, **kwargs: float) -> NDArray[T]:
        """Evaluate the model on ``x`` (and optional ``y``) with named parameters."""
        ...

    @staticmethod
    def guess_initial(x: NDArray, y: NDArray, **kwargs: float | int | str) -> dict[str, float]:
        """Return initial parameter values keyed by ``parameter_schema`` names."""
        ...

    def normalize_value(self, val: float, norm_ctx: NormalizationContext) -> float:
        """Map a physical parameter value into normalized space."""
        ...

    def denormalize_value(self, val: float, norm_ctx: NormalizationContext) -> float:
        """Map a normalized parameter value back to physical units."""
        ...


class ParametricModel(ParameterNormalizationPolicy, ABC):
    """Abstract parametric model with a name, schema, evaluate, and guess_initial."""

    name: ClassVar[str]
    parameter_schema: ClassVar[tuple[ParameterSpec, ...]]
    independent_vars: ClassVar[tuple[str, ...]] = ("x", "y")

    @staticmethod
    @abstractmethod
    def evaluate(*args, **kwargs) -> NDArray:
        """Evaluate the model; subclasses define the concrete signature."""
        ...

    @staticmethod
    @abstractmethod
    def guess_initial(x: NDArray, y: NDArray, **kwargs: float | int | str) -> dict[str, float]:
        """
        Return initial parameter values keyed by ``parameter_schema`` names.

        Peak models expect ``peak_index`` (and optional ``frac``). Background
        models expect ``start`` / ``stop`` on the full spectrum arrays, with
        optional ``mode`` and ``avg_on``.
        """
        ...


class EvaluationLikeFn(Protocol[T]):
    """Callable that evaluates a model given ``x``, ``y``, and named parameters."""

    def __call__(self, x: NDArray[T], y: NDArray[T] | None, **kwargs: float) -> NDArray[T]:
        """Return model intensities at ``x``."""
        ...


class NormalizationLikeFn(Protocol[T]):
    """Callable that maps a scalar through a :class:`NormalizationContext`."""

    def __call__(self, val: float, norm_ctx: NormalizationContext) -> NDArray[T]:
        """Return the mapped value (kept as NDArray for protocol compatibility)."""
        ...


class BasePeakModel(ParametricModel):
    """Parametric model used as a peak component."""

    @staticmethod
    @abstractmethod
    def evaluate(x: NDArray, y: NDArray | None, **kwargs: float) -> NDArray:
        """Evaluate the peak model on ``x``."""
        ...

    @staticmethod
    @abstractmethod
    def guess_initial(x: NDArray, y: NDArray, **kwargs: float | int | str) -> dict[str, float]:
        """Guess peak parameters; requires ``peak_index``."""
        ...


class BaseBackgroundModel(ParametricModel):
    """Parametric model used as a background component."""

    @staticmethod
    @abstractmethod
    def evaluate(x: NDArray, y: NDArray | None, **kwargs: float) -> NDArray:
        """Evaluate the background model on ``x``."""
        ...

    @staticmethod
    @abstractmethod
    def guess_initial(x: NDArray, y: NDArray, **kwargs: float | int | str) -> dict[str, float]:
        """Guess background parameters; requires ``start`` and ``stop``."""
        ...
