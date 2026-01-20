import numpy as np

from .base_models import ParameterSpec, BasePeakModel, BaseBackgroundModel
from .model_funcs import pvoigt, linear_background, static_shirley_background

from numpy.typing import NDArray


class PseudoVoigtPeakModel(BasePeakModel):
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
    def evaluate(x: NDArray, y: NDArray, amp: float, cen: float, sig: float, frac: float) -> NDArray:
        return pvoigt(x, amp, cen, sig, frac)


class ConstantBackgroundModel(BaseBackgroundModel):
    name = "constant"
    parameter_schema = (ParameterSpec("const", 0.0),)
    is_active = False
    normalization_target_parameters = ("const",)
    use_scale = False
    use_offset = True

    @staticmethod
    def evaluate(x: NDArray, y: NDArray, const: float):
        return np.full_like(x, fill_value=const)


class LinearBackgroundModel(BaseBackgroundModel):
    name = "linear"
    parameter_schema = (
        ParameterSpec("i1", 0.0),
        ParameterSpec("i2", 0.0),
    )
    is_active = False
    normalization_target_parameters = ("i1", "i2")
    use_scale = True
    use_offset = True

    @staticmethod
    def evaluate(x: NDArray, y: NDArray, i1: float, i2: float):
        return linear_background(x, i1=i1, i2=i2)


class ShirleyBackgroundModel(BaseBackgroundModel):
    name = "shirley"
    parameter_schema = (
        ParameterSpec("i1", 0.0),
        ParameterSpec("i2", 0.0),
    )
    is_active = False
    normalization_target_parameters = ("i1", "i2")
    use_scale = True
    use_offset = True

    @staticmethod
    def evaluate(x: NDArray, y: NDArray, i1: float, i2: float):
        return static_shirley_background(x, y, i1=i1, i2=i2)
