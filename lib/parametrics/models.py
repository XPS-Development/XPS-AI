import numpy as np
from .base_models import ParameterSpec, BasePeakModel, BaseBackgroundModel

from .normalization import ParameterNormalizationPolicy
from .model_funcs import pvoigt, linear_background, static_shirley_background

from numpy.typing import NDArray


class PseudoVoigtPeakModel(BasePeakModel, ParameterNormalizationPolicy):
    name = "pseudo-voigt"
    parameter_schema = (
        ParameterSpec(name="amp", default=1, lower=0),
        ParameterSpec(name="cen", default=0),
        ParameterSpec(name="sig", default=1, lower=0),
        ParameterSpec(name="frac", default=1, lower=0, upper=1),
    )
    independent_vars = ("x", "y")
    normalization_target_parameters = ("amp",)
    use_scale = True
    use_offset = False

    @staticmethod
    def evaluate(x: NDArray, y: NDArray, amp: float, cen: float, sig: float, frac: float) -> NDArray:
        return pvoigt(x, amp, cen, sig, frac)


class ConstantBackgroundModel(BaseBackgroundModel, ParameterNormalizationPolicy):
    name = "constant"
    parameter_schema = (ParameterSpec("const", 0.0),)
    is_active = False
    independent_vars = ("x", "y")
    normalization_target_parameters = ("const",)
    use_scale = False
    use_offset = True

    @staticmethod
    def evaluate(x: NDArray, y: NDArray, const: float):
        return np.full_like(x, fill_value=const)


class LinearBackgroundModel(BaseBackgroundModel, ParameterNormalizationPolicy):
    name = "linear"
    parameter_schema = (
        ParameterSpec("i1", 0.0),
        ParameterSpec("i2", 0.0),
    )
    independent_vars = ("x", "y")
    is_active = False
    normalization_target_parameters = ("i1", "i2")
    use_scale = True
    use_offset = True

    @staticmethod
    def evaluate(x: NDArray, y: NDArray, i1: float, i2: float):
        return linear_background(x, i1=i1, i2=i2)


class ShirleyBackgroundModel(BaseBackgroundModel, ParameterNormalizationPolicy):
    name = "shirley"
    parameter_schema = (
        ParameterSpec("i1", 0.0),
        ParameterSpec("i2", 0.0),
    )
    is_active = False
    independent_vars = ("x",)
    normalization_target_parameters = ("i1", "i2")
    use_scale = True
    use_offset = True

    @staticmethod
    def evaluate(x: NDArray, y: NDArray, i1: float, i2: float):
        return static_shirley_background(x, y, i1=i1, i2=i2)
