import numpy as np
from scipy import stats

from core.math_models.models import PseudoVoigtPeakModel


def test_pseudo_voigt_evaluate_shape():
    x = np.linspace(-5, 5, 100)
    y = np.zeros_like(x)

    out = PseudoVoigtPeakModel.evaluate(x=x, y=y, amp=1.0, cen=0.0, sig=1.0, frac=0.5)

    assert out.shape == x.shape
    assert np.all(np.isfinite(out))


from core.math_models.models import ConstantBackgroundModel, LinearBackgroundModel, ShirleyBackgroundModel


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
