import numpy as np

from lib.domain import Peak, Background
from lib.parametrics import PseudoVoigtPeakModel, ConstantBackgroundModel, LinearBackgroundModel


def test_peak_evaluate_returns_array():
    model = PseudoVoigtPeakModel()
    peak = Peak(model=model, region_id="r1", amp=2, cen=0, sig=1, frac=1)

    x = np.linspace(-5, 5, 100)
    y = peak.evaluate(x)

    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape


def test_peak_has_correct_parent_and_prefix():
    model = PseudoVoigtPeakModel()
    peak = Peak(model=model, region_id="region-123")

    assert peak.parent_id == "region-123"
    assert peak.id_.startswith("p")


def test_constant_background_evaluate():
    model = ConstantBackgroundModel()
    bg = Background(model=model, region_id="r1", const=5)

    x = np.linspace(0, 10, 50)
    y = np.zeros_like(x)

    out = bg.evaluate(x, y)

    assert np.all(out == 5)


def test_linear_background_evaluate():
    model = LinearBackgroundModel()
    bg = Background(model=model, region_id="r1", i1=1, i2=2)

    x = np.array([0, 1, 2])
    y = np.zeros_like(x)

    out = bg.evaluate(x, y)

    assert out.shape == x.shape


def test_background_has_correct_prefix():
    model = ConstantBackgroundModel()
    bg = Background(model=model, region_id="r1")

    assert bg.id_.startswith("b")
