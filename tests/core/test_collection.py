import numpy as np
import pytest

from core import Spectrum, Region, Peak, Background, SpectrumCollection
from core.math_models import PseudoVoigtPeakModel, ConstantBackgroundModel


def test_collection_add_and_get():
    col = SpectrumCollection()
    s = Spectrum(x=np.arange(5), y=np.arange(5))

    col.add(s)

    assert col.get(s.id_) is s


def test_collection_add_duplicate_raises():
    col = SpectrumCollection()
    s = Spectrum(x=np.arange(3), y=np.arange(3))

    col.add(s)
    with pytest.raises(KeyError):
        col.add(s)


def test_collection_remove_peak_only():
    col = SpectrumCollection()
    s = Spectrum(x=np.arange(5), y=np.arange(5))
    r = Region(slice_=slice(0, 5), parent_id=s.id_)
    p = Peak(model=PseudoVoigtPeakModel(), region_id=r.id_)

    col.add(s)
    col.add(r)
    col.add(p)

    col.remove(p)

    assert p.id_ not in col.objects_index
    assert r.id_ in col.objects_index
    assert s.id_ in col.objects_index


def test_collection_remove_region_cascades():
    col = SpectrumCollection()
    s = Spectrum(x=np.arange(5), y=np.arange(5))
    r = Region(slice_=slice(0, 5), parent_id=s.id_)
    p = Peak(model=PseudoVoigtPeakModel(), region_id=r.id_)
    b = Background(model=ConstantBackgroundModel(), region_id=r.id_)

    for obj in (s, r, p, b):
        col.add(obj)

    col.remove(r)

    assert r.id_ not in col.objects_index
    assert p.id_ not in col.objects_index
    assert b.id_ not in col.objects_index
    assert s.id_ in col.objects_index


def test_collection_remove_spectrum_cascades():
    col = SpectrumCollection()
    s = Spectrum(x=np.arange(5), y=np.arange(5))
    r = Region(slice_=slice(0, 5), parent_id=s.id_)

    col.add(s)
    col.add(r)

    col.remove(s)

    assert not col.objects_index


def test_collection_get_typed():
    col = SpectrumCollection()
    s = Spectrum(x=np.arange(3), y=np.arange(3))
    col.add(s)

    s2 = col.get_typed(s.id_, Spectrum)
    assert s2 is s

    with pytest.raises(TypeError):
        col.get_typed(s.id_, Region)
