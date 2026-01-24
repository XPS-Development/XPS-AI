import numpy as np
import pytest


from core.math_models.normalization import NormalizationContext, ParameterNormalizationPolicy


def test_normalization_context_from_array():
    arr = np.array([1.0, 3.0, 5.0])
    ctx = NormalizationContext.from_array(arr)

    assert ctx.offset == 1.0
    assert ctx.scale == 4.0


def test_normalization_context_invalid_scale():
    arr = np.array([2.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        NormalizationContext.from_array(arr)


class DummyBothNormPolicy(ParameterNormalizationPolicy):
    normalization_target_parameters = tuple()
    use_offset = True
    use_scale = True


class DummyOffsetNormPolicy(ParameterNormalizationPolicy):
    normalization_target_parameters = tuple()
    use_offset = True
    use_scale = False


class DummyScaleNormPolicy(ParameterNormalizationPolicy):
    normalization_target_parameters = tuple()
    use_offset = False
    use_scale = True


def test_parameter_normalization_and_denormalization_roundtrip():
    init_val = 12
    ctx = NormalizationContext(offset=2, scale=10)

    b_policy = DummyBothNormPolicy()
    norm_val = b_policy.normalize_value(init_val, ctx)
    assert norm_val == 1.0
    denorm_val = b_policy.denormalize_value(norm_val, ctx)
    assert denorm_val == init_val

    o_policy = DummyOffsetNormPolicy()
    norm_val = o_policy.normalize_value(init_val, ctx)
    assert norm_val == 10
    denorm_val = o_policy.denormalize_value(norm_val, ctx)
    assert denorm_val == init_val

    s_policy = DummyScaleNormPolicy()
    norm_val = s_policy.normalize_value(init_val, ctx)
    assert norm_val == 1.2
    denorm_val = s_policy.denormalize_value(norm_val, ctx)
    assert denorm_val == init_val
