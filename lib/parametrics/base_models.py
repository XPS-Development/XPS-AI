from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np

from .normalization import BaseNormalizationPolicy

from typing import Optional, Tuple
from numpy.typing import NDArray

T = TypeVar("T", bound=np.floating)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    default: float
    lower: float = -np.inf
    upper: float = np.inf
    vary: bool = True
    expr: Optional[str] = None


class ParametricModel(BaseNormalizationPolicy, ABC):
    name: str
    parameter_schema: Tuple[ParameterSpec, ...]
    independent_vars: Tuple[str, ...]

    @staticmethod
    @abstractmethod
    def evaluate(*args, **kwargs) -> NDArray: ...


class EvaluateFn(Protocol[T]):
    def __call__(self, x: NDArray[T], y: NDArray[T], **kwargs: float) -> NDArray[T]: ...


class ParametricModelLike(Protocol[T]):
    name: str
    parameter_schema: Tuple[str, ...]
    independent_vars: Tuple[str, ...]

    @staticmethod
    def evaluate(x: NDArray[T], y: Optional[NDArray[T]], **kwargs: float) -> NDArray[T]: ...


class BasePeakModel(ParametricModel):
    @staticmethod
    @abstractmethod
    def evaluate(x: NDArray, y: Optional[NDArray], **kwargs: float) -> NDArray: ...


class BaseBackgroundModel(ParametricModel):
    is_active: bool

    @staticmethod
    @abstractmethod
    def evaluate(x: NDArray, y: Optional[NDArray], **kwargs: float) -> NDArray: ...
