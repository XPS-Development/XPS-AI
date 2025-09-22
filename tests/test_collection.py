import numpy as np
import pytest

from lib.spectra import Spectrum, Region, Peak, SpectrumCollection


@pytest.fixture
def spectrum_with_region_and_peak():
    """Создает коллекцию + спектр с регионом и пиком."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    spec = Spectrum(x=x, y=y, name="spec1")

    region = Region(spectrum_id=spec.id)
    peak = Peak(region_id=region.id)

    region.add_peak(peak)
    spec.add_region(region)

    return spec, region, peak


def test_register_objects(spectrum_with_region_and_peak):
    collection = SpectrumCollection()

    spec, region, peak = spectrum_with_region_and_peak

    collection.register(spec)
    collection.register(region)
    collection.register(peak)

    assert spec.id in collection.spectra_index
    assert region.id in collection.region_index
    assert peak.id in collection.peaks_index

    assert collection.get_spectrum(spec.id) is spec
    assert collection.get_region(region.id) is region
    assert collection.get_peak(peak.id) is peak


def test_add_spectrum_registers_nested_objects(spectrum_with_region_and_peak):
    collection = SpectrumCollection()
    spec, region, peak = spectrum_with_region_and_peak

    collection.add_spectrum(spec)

    assert spec.id in collection.spectra_index
    assert region.id in collection.region_index
    assert peak.id in collection.peaks_index


def test_remove_region_updates_collection(spectrum_with_region_and_peak):
    collection = SpectrumCollection()
    spec, region, peak = spectrum_with_region_and_peak
    spec.collection = collection
    region.collection = collection
    spec.add_region(region)  # заново, теперь с коллекцией
    region.add_peak(peak)

    collection.add_spectrum(spec)
    assert region.id in collection.region_index
    assert peak.id in collection.peaks_index

    spec.remove_region(region)

    assert region.id not in collection.region_index
    assert peak.id not in collection.peaks_index


def test_remove_peak_updates_collection(spectrum_with_region_and_peak):
    collection = SpectrumCollection()
    spec, region, peak = spectrum_with_region_and_peak
    spec.collection = collection
    region.collection = collection
    spec.add_region(region)
    region.add_peak(peak)

    collection.add_spectrum(spec)
    assert peak.id in collection.peaks_index

    region.remove_peak(peak)

    assert peak.id not in collection.peaks_index
