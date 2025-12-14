import numpy as np
import pytest

from lib.spectra import Spectrum, Region, Peak, SpectrumCollection


@pytest.fixture
def spectrum_region_peak():
    """Создает спектр, регион и пик отдельно."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    spec = Spectrum(x=x, y=y, name="spec1")

    region = Region()
    peak = Peak()

    return spec, region, peak


def test_register_objects(spectrum_region_peak):
    collection = SpectrumCollection()

    spec, region, peak = spectrum_region_peak

    collection.register(spec)
    collection.register(region)
    collection.register(peak)

    assert spec.id in collection.spectra_index
    assert region.id in collection.regions_index
    assert peak.id in collection.peaks_index

    assert collection.get(spec.id) is spec
    assert collection.get(region.id) is region
    assert collection.get(peak.id) is peak


def test_links(spectrum_region_peak):
    collection = SpectrumCollection()
    spec, region, peak = spectrum_region_peak

    collection.register(spec)
    collection.add_link(spec, region)

    assert region.id in collection.regions_index
    assert region.spectrum_id == spec.id
    assert region.id in spec.regions

    collection.add_link(region, peak)

    assert peak.id in collection.peaks_index
    assert peak.region_id == region.id
    assert peak.id in region.peaks


def test_adding_wrong_links(spectrum_region_peak):
    collection = SpectrumCollection()
    spec, region, peak = spectrum_region_peak

    collection.register(spec)

    with pytest.raises(TypeError):
        collection.add_link(spec, peak)

    with pytest.raises(KeyError):
        collection.add_link(region, spec)


def test_remove_peak(spectrum_region_peak):
    collection = SpectrumCollection()
    spec, region, peak = spectrum_region_peak

    collection.register(spec)
    collection.add_link(spec, region)
    collection.add_link(region, peak)

    collection.remove(peak)

    assert spec.id in collection.spectra_index
    assert region.id in collection.regions_index
    assert peak.id not in collection.peaks_index
    assert peak.id not in region.peaks
    assert peak.region_id is None

    collection.add_link(region, peak)
    collection.remove(peak.id)

    assert spec.id in collection.spectra_index
    assert region.id in collection.regions_index
    assert peak.id not in collection.peaks_index
    assert peak.id not in region.peaks
    assert peak.region_id is None


def test_remove_region(spectrum_region_peak):
    collection = SpectrumCollection()
    spec, region, peak = spectrum_region_peak

    collection.register(spec)
    collection.add_link(spec, region)
    collection.add_link(region, peak)

    collection.remove(region)

    assert spec.id in collection.spectra_index
    assert region.id not in collection.regions_index
    assert peak.id not in collection.peaks_index
    assert region.id not in spec.regions
    assert region.spectrum_id is None

    collection.add_link(spec, region)
    collection.add_link(region, peak)
    collection.remove(region.id)

    assert spec.id in collection.spectra_index
    assert region.id not in collection.regions_index
    assert peak.id not in collection.peaks_index
    assert region.id not in spec.regions
    assert region.spectrum_id is None
