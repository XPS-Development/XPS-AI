import pytest

from core import Spectrum, Region, Background

from services.core import (
    CollectionQueryService,
)

from core.math_models import ModelRegistry


def test_get_parent_region(query_service, simple_collection):
    spectrum_id = next(
        obj.id_ for obj in simple_collection.objects_index.values() if isinstance(obj, Spectrum)
    )
    region_id = query_service.get_regions(spectrum_id)[0]

    assert query_service.get_parent(region_id) == spectrum_id


def test_get_parent_component(query_service):
    spectrum_id = query_service.get_all_spectra()[0]
    region_id = query_service.get_regions(spectrum_id)[0]
    component_id = query_service.get_components(region_id)[0]

    assert query_service.get_parent(component_id) == region_id


def test_get_regions(query_service):
    spectrum_id = query_service.get_all_spectra()[0]
    regions = query_service.get_regions(spectrum_id)

    assert isinstance(regions, tuple)
    assert len(regions) == 1


def test_get_regions_empty(empty_collection):
    srv = CollectionQueryService(empty_collection)
    assert srv.get_all_spectra() == ()


def test_get_components(query_service):
    spectrum_id = query_service.get_all_spectra()[0]
    region_id = query_service.get_regions(spectrum_id)[0]

    components = query_service.get_components(region_id)

    assert all(isinstance(c, str) for c in components)
    assert len(components) >= 1


def test_get_peaks(query_service):
    spectrum_id = query_service.get_all_spectra()[0]
    region_id = query_service.get_regions(spectrum_id)[0]

    peaks = query_service.get_peaks(region_id)

    assert isinstance(peaks, tuple)
    assert len(peaks) == 1


def test_get_background(query_service):
    spectrum_id = query_service.get_all_spectra()[0]
    region_id = query_service.get_regions(spectrum_id)[0]

    bg_id = query_service.get_background(region_id)

    assert isinstance(bg_id, str)


def test_get_background_missing(simple_collection):
    # удаляем background вручную
    bg = next(obj for obj in simple_collection.objects_index.values() if isinstance(obj, Background))
    simple_collection.remove(bg.id_)

    srv = CollectionQueryService(simple_collection)

    spectrum_id = srv.get_all_spectra()[0]
    region_id = srv.get_regions(spectrum_id)[0]

    with pytest.raises(RuntimeError, match="has no Background"):
        srv.get_background(region_id)


def test_get_background_multiple(simple_collection):
    from core.math_models import ModelRegistry

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


def test_get_all_peaks(query_service):
    peaks = query_service.get_all_peaks()

    assert isinstance(peaks, tuple)
    assert len(peaks) >= 1


def test_get_all_spectra(query_service):
    spectra = query_service.get_all_spectra()

    assert spectra
    assert all(isinstance(sid, str) for sid in spectra)


def test_get_all_regions(query_service):
    regions = query_service.get_all_regions()

    assert isinstance(regions, tuple)
    assert len(regions) >= 1


def test_query_service_is_read_only(query_service):
    spectra_before = query_service.get_all_spectra()
    regions_before = query_service.get_all_regions()
    peaks_before = query_service.get_all_peaks()

    spectra_after = query_service.get_all_spectra()
    regions_after = query_service.get_all_regions()
    peaks_after = query_service.get_all_peaks()

    assert spectra_before == spectra_after
    assert regions_before == regions_after
    assert peaks_before == peaks_after
