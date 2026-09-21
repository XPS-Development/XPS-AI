"""Offset/scale context used to normalize and denormalize model parameters."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


@dataclass
class NormalizationContext:
    """Offset and scale derived from an intensity array."""

    offset: float
    scale: float

    @classmethod
    def from_array(cls, arr: NDArray) -> "NormalizationContext":
        """Build a context from the min/max of ``arr``."""
        mn = float(np.min(arr))
        mx = float(np.max(arr))
        scale = mx - mn

        if scale <= 0:
            raise ValueError("Invalid spectrum: zero or negative scale")

        return cls(offset=mn, scale=scale)


class ParameterNormalizationPolicy:
    """Mixin that maps parameter values through a :class:`NormalizationContext`."""

    normalization_target_parameters: ClassVar[tuple[str, ...]] = tuple()
    use_offset: ClassVar[bool] = True
    use_scale: ClassVar[bool] = True

    def normalize_value(self, val: float, norm_ctx: NormalizationContext) -> float:
        """Map a physical value into normalized units."""
        if self.use_offset:
            val -= norm_ctx.offset
        if self.use_scale:
            val /= norm_ctx.scale
        return val

    def denormalize_value(self, val: float, norm_ctx: NormalizationContext) -> float:
        """Map a normalized value back to physical units."""
        if self.use_scale:
            val *= norm_ctx.scale
        if self.use_offset:
            val += norm_ctx.offset
        return val
