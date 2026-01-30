import numpy as np
import pytest

from services.core import SpectrumService
from core import Spectrum
from core.math_models import NormalizationContext


def test_create_spectrum(spectrum_service):
    x = np.linspace(0, 10, 100)
    y = np.random.rand(100)

    sid = spectrum_service.create_spectrum(x, y)

    spectrum = spectrum_service.collection.get_typed(sid, Spectrum)

    assert spectrum.id_ == sid
    assert np.allclose(spectrum.x, x)
    assert np.allclose(spectrum.y, y)
    assert isinstance(spectrum.norm_ctx, NormalizationContext)


def test_create_spectrum_with_explicit_id(simple_gauss_spectrum, spectrum_service):
    x, y = simple_gauss_spectrum

    sid = spectrum_service.create_spectrum(x, y, spectrum_id="spec-1")

    assert sid == "spec-1"


def test_replace_data_updates_arrays_and_norm_ctx(simple_gauss_spectrum, spectrum_service):
    x, y = simple_gauss_spectrum

    sid = spectrum_service.create_spectrum(x, y)

    new_x = x + 1
    new_y = y + 1

    spectrum_service.replace_data(sid, new_x, new_y)

    spectrum = spectrum_service.collection.get_typed(sid, Spectrum)

    assert np.allclose(spectrum.x, new_x)
    assert np.allclose(spectrum.y, new_y)
    assert spectrum.norm_ctx.offset == pytest.approx(new_y.min())


def test_remove_spectrum_removes_object(simple_gauss_spectrum, spectrum_service):
    x, y = simple_gauss_spectrum

    sid = spectrum_service.create_spectrum(x, y)
    spectrum_service.remove_spectrum(sid)

    with pytest.raises(KeyError):
        spectrum_service.collection.get(sid)
