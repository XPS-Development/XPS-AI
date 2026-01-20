from dataclasses import dataclass
import numpy as np

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


class ParameterNormalizationPolicy:
    normalization_target_parameters = tuple[str, ...]
    use_offset = True
    use_scale = True

    def normalize_value(self, val: float, norm_ctx: NormalizationContext) -> float:
        if self.use_offset:
            val -= norm_ctx.offset
        if self.use_scale:
            val /= norm_ctx.scale
        return val

    def denormalize_value(self, val: float, norm_ctx: NormalizationContext) -> float:
        if self.use_scale:
            val *= norm_ctx.scale
        if self.use_offset:
            val += norm_ctx.offset
        return val
