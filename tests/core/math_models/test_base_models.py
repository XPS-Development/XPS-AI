import inspect

from core.math_models.base_models import (
    BaseBackgroundModel,
    BasePeakModel,
    ParameterSpec,
    ParametricModel,
)
from core.math_models.models import (
    ConstantBackgroundModel,
    LinearBackgroundModel,
    PseudoVoigtPeakModel,
    ShirleyBackgroundModel,
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


def test_constant_background_schema_all_inactive_for_optimization():
    """Constant model is fully fixed in schema (optimization subtracts its y contribution)."""
    assert all(not s.vary for s in ConstantBackgroundModel.parameter_schema)


def test_area_comes_from_the_base_model():
    """Peaks report ``amp``; models without an area parameter report None."""
    peak = PseudoVoigtPeakModel()
    assert peak.area({"amp": 2.5, "cen": 0.0, "sig": 1.0, "frac": 0.5}) == 2.5
    assert peak.area({"cen": 0.0}) is None
    assert peak.area({"amp": float("nan")}) is None
    assert ConstantBackgroundModel().area({"const": 1.0}) is None


def test_linear_background_has_mixed_vary_shirley_all_fixed():
    """Linear keeps a varying parameter; Shirley endpoints are fixed (static Shirley)."""
    assert any(s.vary for s in LinearBackgroundModel.parameter_schema)
    assert all(not s.vary for s in ShirleyBackgroundModel.parameter_schema)
