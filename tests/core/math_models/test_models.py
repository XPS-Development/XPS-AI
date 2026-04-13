import numpy as np
from scipy import stats

from core.math_models.models import PseudoVoigtPeakModel
from core.math_models.models import ConstantBackgroundModel, LinearBackgroundModel, ShirleyBackgroundModel
from core.math_models.normalization import NormalizationContext


def run_normalization(model, parameters: dict, ctx: NormalizationContext):
    new_parameters = {}
    for p, v in parameters.items():
        if p in model.normalization_target_parameters:
            v = model.normalize_value(v, ctx)
        new_parameters.update({p: v})
    return new_parameters


def test_pseudo_voigt_evaluate_shape():
    x = np.linspace(-5, 5, 100)
    y = np.zeros_like(x)

    out = PseudoVoigtPeakModel.evaluate(x=x, y=y, amp=1.0, cen=0.0, sig=1.0, frac=0.5)

    assert out.shape == x.shape
    assert np.all(np.isfinite(out))


def test_pseudo_voigt_evaluate_shape_normalization():
    model = PseudoVoigtPeakModel()
    init_parameters = dict(amp=2.0, cen=0.0, sig=1.0, frac=0.5)

    x = np.linspace(-10, 10, 200)
    y = np.zeros_like(x)
    # background
    y += 3

    y += model.evaluate(x, y, **init_parameters)

    ctx = NormalizationContext.from_array(y)

    norm_parameters = run_normalization(model, init_parameters, ctx)

    norm_y = (y - ctx.offset) / ctx.scale
    test_norm_y = model.evaluate(x, y, **norm_parameters)
    assert norm_y.shape == test_norm_y.shape
    assert np.allclose(norm_y, test_norm_y, atol=1e-2)


def test_constant_background():
    x = np.linspace(0, 10, 50)
    y = np.random.rand(50)

    out = ConstantBackgroundModel.evaluate(x, y, const=3.0)

    assert out.shape == x.shape
    assert np.all(out == 3.0)


def test_linear_background_shape():
    x = np.linspace(0, 10, 5)
    bg = LinearBackgroundModel.evaluate(x, y=None, i1=0, i2=10)
    # linear interpolation: bg[0]=0, bg[-1]=10
    assert np.allclose(bg[0], 0)
    assert np.allclose(bg[-1], 10)
    # Should be strictly increasing
    assert np.all(np.diff(bg) > 0)


def test_shirley_background_shape():
    x = np.linspace(-10, 10, 100)
    y = np.exp(-(x**2))
    true_bg = 0.3 * stats.norm(loc=0, scale=np.sqrt(2) / 2).cdf(x) + 10
    y += true_bg

    bg = ShirleyBackgroundModel.evaluate(x, y, i1=y[0], i2=y[-1])

    assert bg.shape == x.shape
    assert np.all(np.isfinite(bg))
    assert np.all(bg >= min(y[0], y[-1]) - 1e-6)
    assert np.all(bg <= y.max() + 1e-6)

    assert np.allclose(bg, true_bg, atol=1e-3)
