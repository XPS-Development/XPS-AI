"""Concrete peak and background models registered with :class:`ModelRegistry`."""

import numpy as np
from numpy.typing import NDArray

from .base_models import BaseBackgroundModel, BasePeakModel, ParameterSpec
from .model_funcs import linear_background, pvoigt, static_shirley_background


class PseudoVoigtPeakModel(BasePeakModel):
    """Pseudo-Voigt peak: linear mix of Gaussian and Lorentzian profiles."""

    name = "pseudo-voigt"
    parameter_schema = (
        ParameterSpec(name="amp", default=1, lower=0),
        ParameterSpec(name="cen", default=0),
        ParameterSpec(name="sig", default=1, lower=0),
        ParameterSpec(name="frac", default=1, lower=0, upper=1),
    )
    normalization_target_parameters = ("amp",)
    use_scale = True
    use_offset = False

    @staticmethod
    def evaluate(
        x: NDArray, y: NDArray, amp: float, cen: float, sig: float, frac: float
    ) -> NDArray:
        """Return the pseudo-Voigt intensity at ``x``."""
        return pvoigt(x, amp, cen, sig, frac)


class ConstantBackgroundModel(BaseBackgroundModel):
    """Constant (flat) background."""

    name = "constant"
    parameter_schema = (ParameterSpec("const", 0.0, vary=False),)
    normalization_target_parameters = ("const",)
    use_scale = False
    use_offset = True

    @staticmethod
    def evaluate(x: NDArray, y: NDArray, const: float) -> NDArray:
        """Return a constant array filled with ``const``."""
        return np.full_like(x, fill_value=const)


class LinearBackgroundModel(BaseBackgroundModel):
    """Linear background between endpoint intensities ``i1`` and ``i2``."""

    name = "linear"
    parameter_schema = (
        ParameterSpec("i1", 0.0, vary=False),
        ParameterSpec("i2", 0.0),
    )
    normalization_target_parameters = ("i1", "i2")
    use_scale = True
    use_offset = True

    @staticmethod
    def evaluate(x: NDArray, y: NDArray, i1: float, i2: float) -> NDArray:
        """Return a linear background spanning ``i1`` to ``i2``."""
        return linear_background(x, i1=i1, i2=i2)


class ShirleyBackgroundModel(BaseBackgroundModel):
    """Iterative Shirley background using endpoint intensities ``i1`` and ``i2``."""

    name = "shirley"
    parameter_schema = (
        ParameterSpec("i1", 0.0, vary=False),
        ParameterSpec("i2", 0.0, vary=False),
    )
    normalization_target_parameters = ("i1", "i2")
    use_scale = True
    use_offset = True

    @staticmethod
    def evaluate(x: NDArray, y: NDArray, i1: float, i2: float) -> NDArray:
        """Return an iterative Shirley background on ``x`` and ``y``."""
        return static_shirley_background(x, y, i1=i1, i2=i2)
