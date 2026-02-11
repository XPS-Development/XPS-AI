import pytest

from core.objects import Region, Background
from core.services import (
    CollectionQueryService,
)
from core.math_models import ModelRegistry


@pytest.fixture
def srv(simple_collection):
    return CollectionQueryService(simple_collection)


def test_get_parent_region(srv, spectrum_id):
    region_id = srv.get_regions(spectrum_id)[0]

    assert srv.get_parent(region_id) == spectrum_id


def test_get_parent_component(srv):
    spectrum_id = srv.get_all_spectra()[0]
    region_id = srv.get_regions(spectrum_id)[0]
    component_id = srv.get_components(region_id)[0]

    assert srv.get_parent(component_id) == region_id


def test_get_regions(srv):
    spectrum_id = srv.get_all_spectra()[0]
    regions = srv.get_regions(spectrum_id)

    assert isinstance(regions, tuple)
    assert len(regions) == 1


def test_get_regions_empty(empty_collection):
    srv = CollectionQueryService(empty_collection)
    assert srv.get_all_spectra() == ()


def test_get_components(srv):
    spectrum_id = srv.get_all_spectra()[0]
    region_id = srv.get_regions(spectrum_id)[0]

    components = srv.get_components(region_id)

    assert all(isinstance(c, str) for c in components)
    assert len(components) >= 1


def test_get_peaks(srv):
    spectrum_id = srv.get_all_spectra()[0]
    region_id = srv.get_regions(spectrum_id)[0]

    peaks = srv.get_peaks(region_id)

    assert isinstance(peaks, tuple)
    assert len(peaks) == 1


def test_get_background(srv):
    spectrum_id = srv.get_all_spectra()[0]
    region_id = srv.get_regions(spectrum_id)[0]

    bg_id = srv.get_background(region_id)

    assert isinstance(bg_id, str)


def test_get_background_multiple(simple_collection):

    region_id = next(obj.id_ for obj in simple_collection.objects_index.values() if isinstance(obj, Region))

    model = ModelRegistry.get("constant")

    bg2 = Background(
        model=model,
        region_id=region_id,
    )
    simple_collection.add(bg2)

    srv = CollectionQueryService(simple_collection)

    with pytest.raises(RuntimeError, match="multiple Backgrounds"):
        srv.get_background(region_id)


def test_get_all_peaks(srv):
    peaks = srv.get_all_peaks()

    assert isinstance(peaks, tuple)
    assert len(peaks) >= 1


def test_get_all_spectra(srv):
    spectra = srv.get_all_spectra()

    assert spectra
    assert all(isinstance(sid, str) for sid in spectra)


def test_get_all_regions(srv):
    regions = srv.get_all_regions()

    assert isinstance(regions, tuple)
    assert len(regions) >= 1


def test_query_service_is_read_only(srv):
    spectra_before = srv.get_all_spectra()
    regions_before = srv.get_all_regions()
    peaks_before = srv.get_all_peaks()

    spectra_after = srv.get_all_spectra()
    regions_after = srv.get_all_regions()
    peaks_after = srv.get_all_peaks()

    assert spectra_before == spectra_after
    assert regions_before == regions_after
    assert peaks_before == peaks_after
