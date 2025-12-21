import numpy as np
import pytest

from lib.domain import Spectrum


def test_spectrum_creates_norm_context():
    x = np.linspace(0, 10, 5)
    y = np.array([1, 2, 3, 4, 5])

    s = Spectrum(x=x, y=y)

    assert s.norm_ctx.offset == 1
    assert s.norm_ctx.scale == 4


def test_spectrum_creates_id():
    s = Spectrum(x=np.arange(3), y=np.arange(3))
    assert s.id_.startswith("s")


def test_spectrum_validates_shape():
    with pytest.raises(ValueError):
        Spectrum(x=np.arange(3), y=np.arange(4))


def test_spectrum_validates_ndim():
    with pytest.raises(ValueError):
        Spectrum(x=np.ones((3, 1)), y=np.ones((3,)))


def test_spectrum_parent_id_is_none():
    s = Spectrum(x=np.arange(3), y=np.arange(3))
    assert s.parent_id is None
