import inspect

from lib.parametrics.base_models import (
    ParametricModel,
    BasePeakModel,
    BaseBackgroundModel,
    ParameterSpec,
)


def test_parameter_spec_defaults():
    spec = ParameterSpec(name="a", default=1.0)
    assert spec.lower == float("-inf")
    assert spec.upper == float("inf")
    assert spec.vary is True
    assert spec.expr is None


def test_parametric_model_contract():
    # Abstract base: cannot instantiate
    assert inspect.isabstract(ParametricModel)
    assert inspect.isabstract(BasePeakModel)
    assert inspect.isabstract(BaseBackgroundModel)
