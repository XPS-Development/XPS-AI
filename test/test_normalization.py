import numpy as np
import pytest

from lib.parametrics.normalization import (
    NormalizationContext,
    RuntimeParameter,
    ParameterNormalizationPolicy,
)


def test_normalization_context_from_array():
    arr = np.array([1.0, 3.0, 5.0])
    ctx = NormalizationContext.from_array(arr)

    assert ctx.offset == 1.0
    assert ctx.scale == 4.0


def test_normalization_context_invalid_scale():
    arr = np.array([2.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        NormalizationContext.from_array(arr)


class DummyNormPolicy(ParameterNormalizationPolicy):
    normalization_target_parameters = ("a",)


def test_parameter_normalization_and_denormalization_roundtrip():
    params = {
        "a": RuntimeParameter("a", value=10, lower=0, upper=20),
        "b": RuntimeParameter("b", value=5),
    }

    ctx = NormalizationContext(offset=0, scale=10)
    policy = DummyNormPolicy()

    norm = policy.normalize(params, ctx)
    assert norm["a"].value == 1.0
    assert norm["b"] is params["b"]

    denorm = policy.denormalize(norm, ctx)
    assert denorm["a"].value == 10
