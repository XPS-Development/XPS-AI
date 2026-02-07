import numpy as np
import pytest

from core.services import SpectrumService
from core.objects import Spectrum
from core.math_models import NormalizationContext


@pytest.fixture
def srv(simple_collection):
    return SpectrumService(simple_collection)


def test_create_spectrum(srv):
    x = np.linspace(0, 10, 100)
    y = np.random.rand(100)

    sid = srv.create_spectrum(x, y)

    spectrum = srv.collection.get_typed(sid, Spectrum)

    assert spectrum.id_ == sid
    assert np.allclose(spectrum.x, x)
    assert np.allclose(spectrum.y, y)
    assert isinstance(spectrum.norm_ctx, NormalizationContext)


def test_create_spectrum_with_explicit_id(simple_gauss_spectrum, srv):
    x, y = simple_gauss_spectrum

    sid = srv.create_spectrum(x, y, spectrum_id="spec-1")

    assert sid == "spec-1"


def test_replace_data_updates_arrays_and_norm_ctx(simple_gauss_spectrum, srv):
    x, y = simple_gauss_spectrum

    sid = srv.create_spectrum(x, y)

    new_x = x + 1
    new_y = y + 1

    srv.replace_data(sid, new_x, new_y)

    spectrum = srv.collection.get_typed(sid, Spectrum)

    assert np.allclose(spectrum.x, new_x)
    assert np.allclose(spectrum.y, new_y)
    assert spectrum.norm_ctx.offset == pytest.approx(new_y.min())


def test_remove_spectrum_removes_object(simple_gauss_spectrum, srv):
    x, y = simple_gauss_spectrum

    sid = srv.create_spectrum(x, y)
    srv.remove_spectrum(sid)

    with pytest.raises(KeyError):
        srv.collection.get(sid)
