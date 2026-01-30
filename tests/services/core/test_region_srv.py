import numpy as np
import pytest

from services.core import RegionService
from core import Region, Spectrum


def test_create_region(region_service, spectrum_id):
    rid = region_service.create_region(spectrum_id, start=0, stop=10)

    region = region_service.collection.get_typed(rid, Region)

    assert region.parent_id == spectrum_id
    assert region.slice_ == slice(0, 10)


@pytest.mark.parametrize(
    "start, stop",
    [
        (-1, 5),
        (0, 1000),
    ],
)
def test_create_region_out_of_bounds(region_service, spectrum_id, start, stop):
    with pytest.raises(IndexError):
        region_service.create_region(spectrum_id, start, stop)


def test_create_region_invalid_slice(region_service, spectrum_id):
    with pytest.raises(ValueError):
        region_service.create_region(spectrum_id, start=5, stop=5)


def test_update_slice(region_service):
    region = next(obj for obj in region_service.collection.objects_index.values() if isinstance(obj, Region))

    region_service.update_slice(region.id_, start=2, stop=8)

    assert region.slice_ == slice(2, 8)


def test_update_slice_invalid(region_service):
    region = next(obj for obj in region_service.collection.objects_index.values() if isinstance(obj, Region))

    with pytest.raises(ValueError):
        region_service.update_slice(region.id_, start=-1, stop=5)


def test_remove_region(region_service):
    region = next(obj for obj in region_service.collection.objects_index.values() if isinstance(obj, Region))

    region_service.remove_region(region.id_)

    with pytest.raises(KeyError):
        region_service.collection.get(region.id_)
