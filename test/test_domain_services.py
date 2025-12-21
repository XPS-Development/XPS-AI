import numpy as np
import pytest

from lib.domain import (
    SpectrumCollection,
    Spectrum,
    Region,
    Peak,
    Background,
)
from lib.domain_services import (
    CollectionQueryService,
    SpectrumService,
    RegionService,
    ComponentService,
    DataQueryService,
)
from lib.parametrics import ModelRegistry


@pytest.fixture
def collection():
    return SpectrumCollection()


@pytest.fixture
def spectrum_data():
    x = np.linspace(-10, 10, 100)
    y = np.exp(-(x**2)) + 2
    return x, y


@pytest.fixture
def spectrum(collection, spectrum_data):
    svc = SpectrumService(collection)
    x, y = spectrum_data
    spectrum_id = svc.create_spectrum(x, y)
    return spectrum_id


@pytest.fixture
def region(collection, spectrum):
    svc = RegionService(collection)
    return svc.create_region(spectrum, start=10, stop=50)


def test_get_all_spectra(collection, spectrum):
    qs = CollectionQueryService(collection)

    spectra = qs.get_all_spectra()

    assert len(spectra) == 1
    assert spectra[0].id_ == spectrum


def test_get_regions(collection, spectrum):
    rs = RegionService(collection)
    region_id = rs.create_region(spectrum, 0, 10)

    qs = CollectionQueryService(collection)
    regions = qs.get_regions(spectrum)

    assert len(regions) == 1
    assert regions[0].id_ == region_id


def test_get_background_errors(collection, region):
    qs = CollectionQueryService(collection)

    with pytest.raises(RuntimeError, match="has no Background"):
        qs.get_background(region)


def test_create_spectrum_registers_object(collection, spectrum_data):
    svc = SpectrumService(collection)
    x, y = spectrum_data

    spectrum_id = svc.create_spectrum(x, y)

    spectrum = collection.get(spectrum_id)
    assert isinstance(spectrum, Spectrum)


def test_replace_data_updates_norm_ctx(collection, spectrum):
    svc = SpectrumService(collection)

    x = np.linspace(0, 1, 50)
    y = np.exp(-((x - 5) ** 2)) + 1

    svc.replace_data(spectrum, x, y)

    spectrum_obj = collection.get(spectrum)
    assert spectrum_obj.norm_ctx is not None


def test_remove_spectrum(collection, spectrum):
    svc = SpectrumService(collection)

    svc.remove_spectrum(spectrum)

    with pytest.raises(KeyError):
        collection.get(spectrum)


def test_create_region_valid(collection, spectrum):
    svc = RegionService(collection)

    region_id = svc.create_region(spectrum, 5, 20)

    region = collection.get(region_id)
    assert isinstance(region, Region)
    assert region.parent_id == spectrum


def test_create_region_invalid_slice(collection, spectrum):
    svc = RegionService(collection)

    with pytest.raises(ValueError):
        svc.create_region(spectrum, 10, 5)


def test_update_slice(collection, region):
    svc = RegionService(collection)

    svc.update_slice(region, 20, 30)

    region_obj = collection.get(region)
    assert region_obj.slice_.start == 20


def test_create_peak(collection, region):
    svc = ComponentService(collection)

    peak_id = svc.create_peak(
        region_id=region,
        model_name="pseudo-voigt",
        parameters={"amp": 1.0},
    )

    peak = collection.get(peak_id)
    assert isinstance(peak, Peak)
    assert peak.parent_id == region


def test_create_peak_wrong_model(collection, region):
    svc = ComponentService(collection)

    with pytest.raises(ValueError, match="not a peak model"):
        svc.create_peak(region, model_name="linear")


def test_replace_background_creates(collection, region):
    svc = ComponentService(collection)

    bg_id = svc.replace_background(
        region_id=region,
        model_name="constant",
        parameters={"const": 1.0},
    )

    bg = collection.get(bg_id)
    assert isinstance(bg, Background)


def test_replace_background_replaces_existing(collection, region):
    svc = ComponentService(collection)

    bg1 = svc.replace_background(region, "constant")
    bg2 = svc.replace_background(region, "linear")

    with pytest.raises(KeyError):
        collection.get(bg1)

    assert isinstance(collection.get(bg2), Background)


def test_replace_background_multiple_existing_error(collection, region):
    svc = ComponentService(collection)

    bg1 = svc.replace_background(region, "constant")
    bg2 = Background(
        model=ModelRegistry.get("constant"),
        region_id=region,
    )
    collection.add(bg2)

    with pytest.raises(RuntimeError, match="exactly one Background"):
        svc.replace_background(region, "linear_background")


def test_set_and_get_parameter(collection, region):
    svc = ComponentService(collection)

    peak_id = svc.create_peak(region, "pseudo-voigt")
    svc.set_parameter(peak_id, "amp", value=2.0)

    par = svc.get_parameter(peak_id, "amp")
    assert par.value == 2.0


def test_copy_parameters(collection, region):
    svc = ComponentService(collection)

    p1 = svc.create_peak(region, "pseudo-voigt", {"amp": 1.0})
    p2 = svc.create_peak(region, "pseudo-voigt", {"amp": 0.0})

    svc.copy_parameters(p1, p2)

    assert svc.get_parameter(p2, "amp").value == 1.0


def test_copy_parameters_model_mismatch(collection, region):
    svc = ComponentService(collection)

    p = svc.create_peak(region, "pseudo-voigt")
    bg = svc.replace_background(region, "constant")

    with pytest.raises(ValueError, match="does not match"):
        svc.copy_parameters(p, bg)


def test_get_spectrum_data_raw(collection, spectrum):
    svc = DataQueryService(collection)

    x, y = svc.get_spectrum_data(spectrum)

    assert len(x) == len(y)


def test_get_spectrum_data_normalized(collection, spectrum):
    svc = DataQueryService(collection)

    x, y_norm = svc.get_spectrum_data(spectrum, normalized=True)

    assert len(x) == len(y_norm)

    assert y_norm.max() == pytest.approx(1.0, abs=1e-6)
    assert y_norm.min() == pytest.approx(0, abs=1e-6)


def test_get_region_data(collection, region):
    svc = DataQueryService(collection)

    x, y = svc.get_region_data(region)

    assert len(x) == len(y)


def test_get_region_data_normalized(collection, region):
    svc = DataQueryService(collection)
    x, y_norm = svc.get_region_data(region, normalized=True)

    assert len(x) == len(y_norm)

    assert y_norm.max() <= 1
    assert y_norm.min() >= 0
