from dataclasses import dataclass

import numpy as np

from .normalization import ParameterNormalizationPolicy, NormalizationContext

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Protocol, TypeVar
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


class ParametricModelLike(Protocol[T]):
    name: str
    parameter_schema: Tuple[ParameterSpec, ...]
    independent_vars: Tuple[str, ...]
    normalization_target_parameters = Tuple[str, ...]
    use_offset: bool = True
    use_scale: bool = True

    @staticmethod
    def evaluate(x: NDArray[T], y: Optional[NDArray[T]], **kwargs: float) -> NDArray[T]: ...

    def normalize_value(self, val: float, norm_ctx: NormalizationContext) -> float: ...

    def denormalize_value(self, val: float, norm_ctx: NormalizationContext) -> float: ...


class ParametricModel(ParameterNormalizationPolicy, ABC):
    name: str
    parameter_schema: Tuple[ParameterSpec, ...]
    independent_vars: Tuple[str, ...] = ("x", "y")

    @staticmethod
    @abstractmethod
    def evaluate(*args, **kwargs) -> NDArray: ...


class EvaluationLikeFn(Protocol[T]):
    def __call__(self, x: NDArray[T], y: NDArray[T], **kwargs: float) -> NDArray[T]: ...


class NormalizationLikeFn(Protocol[T]):
    def __call__(self, val: float, norm_ctx: NormalizationContext) -> NDArray[T]: ...


class BasePeakModel(ParametricModel):
    @staticmethod
    @abstractmethod
    def evaluate(x: NDArray, y: Optional[NDArray], **kwargs: float) -> NDArray: ...


class BaseBackgroundModel(ParametricModel):
    static: bool

    @staticmethod
    @abstractmethod
    def evaluate(x: NDArray, y: Optional[NDArray], **kwargs: float) -> NDArray: ...
