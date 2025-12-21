from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from .runtime import RuntimeParameter

from typing import Dict, Any
from numpy.typing import NDArray


@dataclass
class NormalizationContext:
    offset: float
    scale: float

    @classmethod
    def from_array(cls, arr: NDArray) -> "NormalizationContext":
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        scale = mx - mn

        if scale <= 0:
            raise ValueError("Invalid spectrum: zero or negative scale")

        return cls(offset=mn, scale=scale)


class BaseNormalizationPolicy(ABC):
    use_offset: bool
    use_scale: bool

    @abstractmethod
    def normalize(self, *args, **kwargs) -> Any: ...

    @abstractmethod
    def denormalize(self, *args, **kwargs) -> Any: ...


class ParameterNormalizationPolicy(BaseNormalizationPolicy):
    normalization_target_parameters = tuple[str, ...]
    use_offset = True
    use_scale = True

    def _normalize_value(self, val: float, norm_ctx: NormalizationContext) -> float:
        if self.use_offset:
            val -= norm_ctx.offset
        if self.use_scale:
            val /= norm_ctx.scale
        return val

    def _denormalize_value(self, val: float, norm_ctx: NormalizationContext) -> float:
        if self.use_scale:
            val *= norm_ctx.scale
        if self.use_offset:
            val += norm_ctx.offset
        return val

    def normalize(
        self, parameters: Dict[str, RuntimeParameter], norm_ctx: NormalizationContext
    ) -> Dict[str, RuntimeParameter]:
        new_params = {}

        for name, p in parameters.items():
            if name in self.normalization_target_parameters:
                new_params[name] = p.copy_with(
                    value=self._normalize_value(p.value, norm_ctx),
                    lower=self._normalize_value(p.lower, norm_ctx),
                    upper=self._normalize_value(p.upper, norm_ctx),
                )
            else:
                new_params[name] = p

        return new_params

    def denormalize(
        self, parameters: Dict[str, RuntimeParameter], norm_ctx: NormalizationContext
    ) -> Dict[str, RuntimeParameter]:
        new_params = {}

        for name, p in parameters.items():
            if name in self.normalization_target_parameters:
                new_params[name] = p.copy_with(
                    value=self._denormalize_value(p.value, norm_ctx),
                    lower=self._denormalize_value(p.lower, norm_ctx),
                    upper=self._denormalize_value(p.upper, norm_ctx),
                )
            else:
                new_params[name] = p

        return new_params
